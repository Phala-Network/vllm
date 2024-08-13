import asyncio
import importlib
import inspect
import re
from argparse import Namespace
from contextlib import asynccontextmanager
from http import HTTPStatus
from multiprocessing import Process
from typing import AsyncIterator, List, Set

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.protocol import AsyncEngineClient
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import make_arg_parser
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              DetokenizeRequest,
                                              DetokenizeResponse,
                                              EmbeddingRequest, ErrorResponse,
                                              TokenizeRequest,
                                              TokenizeResponse)
from vllm.entrypoints.openai.rpc.client import AsyncEngineRPCClient
from vllm.entrypoints.openai.rpc.server import run_rpc_server
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, get_open_port
from vllm.version import __version__ as VLLM_VERSION

import web3, eth_utils, eth_account
from hashlib import sha256
from verifier import cc_admin
import base64, json

TIMEOUT_KEEP_ALIVE = 5  # seconds

chat_0: OpenAIServingChat
chat_1: OpenAIServingChat
chat_2: OpenAIServingChat

signing_address: str
nvidia_payload: str

all_chats = dict[str, object]

logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: Set[asyncio.Task] = set()


def model_is_embedding(model_name: str, trust_remote_code: bool) -> bool:
    return ModelConfig(model=model_name,
                       tokenizer=model_name,
                       tokenizer_mode="auto",
                       trust_remote_code=trust_remote_code,
                       seed=0,
                       dtype="float16").embedding_mode


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


router = APIRouter()


def mount_metrics(app: FastAPI):
    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app())
    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile('^/metrics(?P<path>.*)$')
    app.routes.append(metrics_route)


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request
):
    if request.model.lower() == "meta-llama/meta-llama-3.1-8b-instruct":
        chat = chat_0
    elif request.model.lower() == "google/gemma-2-9b-it":
        chat = chat_1
    else:
        chat = chat_2

    global raw_acct, all_chats
    signing_mode = raw_request.headers.get('x-phala-signature-type')
    generator = await chat.create_chat_completion(
        request, raw_request, raw_acct, signing_mode, all_chats)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump(exclude_none=True))


@router.get("/v1/attestation/report")
async def create_attestation_report():
    return JSONResponse(content={
        "signing_address": signing_address,
        "nvidia_payload": nvidia_payload,
    })


@router.get("/v1/signing/{request_id}")
async def get_signing(request_id):
    global all_chats, raw_acct
    return JSONResponse(content = {
        "text": all_chats[request_id],
        "signature": raw_acct.sign_message(eth_account.messages.encode_defunct(text = all_chats[request_id])).signature.hex()
    })


def build_app(args: Namespace) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        err = chat_0.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if request.method == "OPTIONS":
                return await call_next(request)
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    return app


async def init_app(
    engines: List[AsyncEngineClient],
    args: Namespace,
) -> FastAPI:
    app = build_app(args)

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    global chat_0
    chat_0 = OpenAIServingChat(
        engines[0],
        await engines[0].get_model_config(),
        ["meta-llama/meta-llama-3.1-8b-instruct"],
        args.response_role,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        chat_template=args.chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    )

    global chat_1
    chat_1 = OpenAIServingChat(
        engines[1],
        await engines[1].get_model_config(),
        ["google/gemma-2-9b-it"],
        args.response_role,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        chat_template=args.chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    )

    global chat_2
    chat_2 = OpenAIServingChat(
        engines[2],
        await engines[2].get_model_config(),
        ["microsoft/phi-3-mini-4k-instruct"],
        args.response_role,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        chat_template=args.chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    )

    w3 = web3.Web3()
    global raw_acct
    raw_acct = w3.eth.account.create()
    pub_keccak = eth_utils.keccak(raw_acct._key_obj.public_key.to_bytes()).hex()
    gpu_evidence = cc_admin.collect_gpu_evidence(pub_keccak)[0]

    global signing_address
    signing_address = raw_acct.address

    global nvidia_payload
    nvidia_payload = build_payload(pub_keccak, gpu_evidence['attestationReportHexStr'], gpu_evidence['certChainBase64Encoded'])

    global all_chats
    all_chats = dict()

    app.root_path = args.root_path

    return app


def build_payload(nonce, evidence, cert_chain):
    data = dict()
    data['nonce'] = nonce
    encoded_evidence_bytes = evidence.encode("ascii")
    encoded_evidence = base64.b64encode(encoded_evidence_bytes)
    encoded_evidence = encoded_evidence.decode('utf-8')
    data['evidence'] = encoded_evidence
    data['arch'] = 'HOPPER'
    data['certificate'] = str(cert_chain)
    payload = json.dumps(data)
    return payload


async def run_server(args, args1, args2, **uvicorn_kwargs) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)

    logger.info("args0: %s", args)
    engine_args0 = AsyncEngineArgs.from_cli_args(args)
    engine0 = AsyncLLMEngine.from_engine_args(
        engine_args0,
        usage_context=UsageContext.OPENAI_API_SERVER
    )

    logger.info("args1: %s", args1)
    engine_args1 = AsyncEngineArgs.from_cli_args(args1)
    engine1 = AsyncLLMEngine.from_engine_args(
        engine_args1,
        usage_context=UsageContext.OPENAI_API_SERVER
    )

    logger.info("args2: %s", args2)
    engine_args2 = AsyncEngineArgs.from_cli_args(args2)
    engine2 = AsyncLLMEngine.from_engine_args(
        engine_args2,
        usage_context=UsageContext.OPENAI_API_SERVER
    )

    app = await init_app([engine0, engine1, engine2], args0)

    shutdown_task = await serve_http(
        app,
        host=args.host,
        port=6974,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)

    args0 = parser.parse_args(["--model=/mnt/models/meta-llama/Meta-Llama-3.1-8B-Instruct/", "--gpu-memory-utilization=0.35"])
    args1 = parser.parse_args(["--model=/mnt/models/google/gemma-2-9b-it/", "--enforce-eager", "--gpu-memory-utilization=0.3"])
    args2 = parser.parse_args(["--model=/mnt/models/microsoft/Phi-3-mini-4k-instruct/", "--enforce-eager", "--gpu-memory-utilization=0.1"])

    asyncio.run(run_server(args0, args1, args2))
