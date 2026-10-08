"""Build-time source-guarded insertion; no runtime patching."""
import hashlib
import importlib.util
import json
from pathlib import Path

root = Path(importlib.util.find_spec('vllm').origin).parent
target = root / 'distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py'
source = target.read_bytes()
expected = 'a8fdf8179b8518510d6be7be12c0ba066aed7d6aa0e11a371d02880ce772a4f0'
assert hashlib.sha256(source).hexdigest() == expected, 'Upstream worker SHA256 mismatch'
marker = b'        # Start transfer threads\n'
assert source.count(marker) == 1
insertion = b'        from .runtime_layout_audit import log_layout\n\n        log_layout(self, kv_caches, group_kernel_blocks, logger)\n\n'
patched = source.replace(marker, insertion + marker)
compile(patched, str(target), 'exec')
target.write_bytes(patched)
helper = Path('/opt/runtime-model/layout_audit.py').read_bytes()
(target.parent / 'runtime_layout_audit.py').write_bytes(helper)
Path('/opt/runtime-model/layout-patch-receipt.json').write_text(json.dumps({'upstream_sha256': expected, 'patched_sha256': hashlib.sha256(patched).hexdigest(), 'helper_sha256': hashlib.sha256(helper).hexdigest(), 'scope': 'single registration-time metadata log before transfer threads; no transfer changes'}, indent=2) + '\n')
