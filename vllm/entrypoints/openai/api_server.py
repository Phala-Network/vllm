import asyncio
import importlib
import inspect
import re
from argparse import Namespace
from contextlib import asynccontextmanager
from http import HTTPStatus
from multiprocessing import Process
from typing import AsyncIterator, List, Set, Annotated

from fastapi import APIRouter, FastAPI, Header, Request, Response, Path
from fastapi.encoders import jsonable_encoder
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
                                              ErrorResponse)
from vllm.entrypoints.openai.rpc.client import AsyncEngineRPCClient
from vllm.entrypoints.openai.rpc.server import run_rpc_server
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, get_open_port, random_uuid
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


@router.post("/v1/chat/completions", responses={
    200: {
        "content": {
            "application/json": {
                "example": {
  "id": "chat-ab1aa731c64347558cc8ba1ad8c9ca32",
  "object": "chat.completion",
  "created": 1723517431,
  "model": "meta-llama/meta-llama-3.1-8b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I don't have a personal model name, but I'm a variant of the LLaMA (Large Language Model Application) model, which is a type of artificial intelligence designed to understand and generate human-like text. I'm a helpful assistant, and I'm here to assist you with any questions or tasks you may have!",
        "tool_calls": []
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 27,
    "total_tokens": 93,
    "completion_tokens": 66
  },
  "signature": "0xd9533aab32865f2168189935730858a87ab1cba63d35faef3d8311b5013fa9da4b42c2726444f550406378b1980c34c3ac192dc5159179e7dafc2d90e5ccde7a1c"
                }
            }
        }
    }
},
    summary="Creates a model response for the given that conversation.",
    description="Given a list of messages comprising a conversation, the model will return a response.\n\nCurrently, we support `meta-llama/meta-llama-3.1-8b-instruct`, `google/gemma-2-9b-it` and `microsoft/phi-3-mini-4k-instruct` as the model.")
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
    x_phala_signature_type: Annotated[str | None, Header(description="StandaloneApi or ModifiedResponse. If other value is set, will not have signature", example="ModifiedResponse")] = None
) -> Response:
    if request.model.lower() == "meta-llama/meta-llama-3.1-8b-instruct":
        chat = chat_0
    elif request.model.lower() == "google/gemma-2-9b-it":
        chat = chat_1
    else:
        chat = chat_2

    global raw_acct
    generator = await chat.create_chat_completion(
        request, raw_request, raw_acct, x_phala_signature_type)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if not request.stream:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump(exclude_none=True))
    else:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")


@router.get("/v1/attestation/report", summary="Get the signing key address and create the nvidia attestation payload.", responses={
    200: {
        "content": {
            "application/json": {
                "example": {
                    "signing_address": "...",
                    "nvidia_payload": "..."
                }
            }
        }
    }
})
async def create_attestation_report():
    return JSONResponse(content={
        "signing_address": signing_address,
        "nvidia_payload": nvidia_payload,
    })


@router.get("/v1/signature/{chat_id}", summary="Get the signature for chat_completion if the SignatureType is StandaloneApi", responses={
    200: {
        "content": {
            "application/json": {
                "example": {
                    "text": "[{\"content\": \"You are a helpful assistant.\", \"role\": \"system\"}, {\"content\": \"What is your model name?\", \"role\": \"user\"}]\n[{\"index\": 0, \"message\": {\"role\": \"assistant\", \"content\": \"I don't have a specific model name, as I'm a large language model, I'm a part of a broader AI system. However, I'm a variant of the popular language model, Llama (Large Language Model Meta AI).\", \"tool_calls\": []}, \"logprobs\": null, \"finish_reason\": \"stop\", \"stop_reason\": null}]",
                    "signature": "0xa3de1510a0537897688509a1a7ea78802fd2e8d35c54844b52815720233f016e3bb4853a4c0ac72ee899dd11d91feaf3ad8daa1d4518b4e22cad2a9f57026eee1b"
                }
            }
        }
    }
})
async def get_signature(chat_id: Annotated[str, Path(description="chat_id from the chat_completion", example="chatcmpl-ab1aa731c64347558cc8ba1ad8c9ca32")]):
    global all_chats, raw_acct
    return JSONResponse(content = {
        "text": all_chats[chat_id],
        "signature": raw_acct.sign_message(eth_account.messages.encode_defunct(text = all_chats[chat_id])).signature.hex()
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

    @app.middleware("http")
    async def signature(request: Request, call_next):
        root_path = "" if args.root_path is None else args.root_path
        if request.url.path.startswith(f"{root_path}/v1/chat/completions"):
            h = sha256()
            h.update(await request.body())
            request_sha256 = h.digest().hex()

            response = await call_next(request)

            if hasattr(request.state, 'request_id'):
                request_id = request.state.request_id
            else:
                request_id = f"undefined-{random_uuid()}"
            response.headers["x-phala-request-id"] = request_id

            h = sha256()
            original_iterator = response.body_iterator
            async def new_iterator():
                async for chunk in original_iterator:
                    h.update(chunk)
                    print(chunk)
                    yield chunk
                response_sha256 = h.hexdigest()
                global all_chats
                all_chats[request_id] = request_sha256 + ':' + response_sha256
            response.body_iterator = new_iterator()

            return response
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
