"""Fetch only pinned small artifacts; lock and verify every byte. No weights."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MODELS = {
    "target": ("RedHatAI/Kimi-K3-NVFP4", "13fd4e58db6abe84bd75aecd055b7e18057cf607"),
    "draft": ("RedHatAI/Kimi-K3-speculator.dspark", "38a88101e0d46bb22134b9da340f381b954d40d4"),
}
CODE = {"tokenization_kimi.py", "encoding_k3.py", "kimi_k3_vision_processing.py", "media_utils.py", "kimi_k3_processor.py"}

def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="Create initial lock; refuses to overwrite")
    args = parser.parse_args()
    lock = ROOT / "manifest.json"
    if args.freeze and lock.exists():
        raise SystemExit("Existing manifest must not be silently re-frozen")
    expected = None if args.freeze else json.loads(lock.read_text())
    manifest = {"base_image": "vllm/vllm-openai:v0.31.0@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b", "models": {}}
    for key, (repo, revision) in MODELS.items():
        metadata = json.loads(fetch(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"))
        assert metadata["sha"] == revision
        row = {"repo": repo, "revision": revision, "view": f"/opt/runtime-model/{key}", "weights_root": f"/opt/model-data/{repo.replace('/', '--')}-{revision[:8]}", "files": [], "weights": []}
        for file in metadata["siblings"]:
            name = file["rfilename"]
            assert "/" not in name and "\\" not in name, name
            if name.endswith(".safetensors"):
                row["weights"].append({"name": name, "size": file["size"], "sha256": file["lfs"]["sha256"]})
                continue
            if not (name.endswith(".json") or name == "tiktoken.model" or (key == "target" and name in CODE)):
                continue
            # The NVFP4 index contains per-expert tensors and is ~121 MB.
            assert file["size"] < (160_000_000 if name.endswith('.safetensors.index.json') else 30_000_000), name
            data = fetch(f"https://huggingface.co/{repo}/resolve/{revision}/{name}")
            assert len(data) == file["size"], name
            if "lfs" in file:
                assert hashlib.sha256(data).hexdigest() == file["lfs"]["sha256"], name
            else:
                assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == file["blobId"], name
            row["files"].append({"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            dest = ROOT / "artifacts" / key / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            if name.endswith(".py"):
                ast.parse(data)
        assert row["weights"]
        row["files"].sort(key=lambda r: r["name"])
        row["weights"].sort(key=lambda r: r["name"])
        manifest["models"][key] = row
    if expected is not None:
        assert manifest == expected, "Pinned artifact manifest mismatch"
    else:
        lock.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {"files": len(v["files"]), "weight_links": len(v["weights"])} for k, v in manifest["models"].items()}))

if __name__ == "__main__":
    main()
