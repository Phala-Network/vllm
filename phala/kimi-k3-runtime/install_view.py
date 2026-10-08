"""Build-time only: check baked bytes and create links for weights, never code."""
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path('/opt/runtime-model/manifest.json').read_text())
for row in manifest['models'].values():
    root = Path(row['view'])
    for file in row['files']:
        path = root / file['name']
        data = path.read_bytes()
        assert len(data) == file['size'] and hashlib.sha256(data).hexdigest() == file['sha256'], path
    for weight in row['weights']:
        (root / weight['name']).symlink_to(Path(row['weights_root']) / weight['name'])
    actual = {p.name for p in root.iterdir()}
    assert actual == {f['name'] for f in row['files'] + row['weights']}, actual
print('Immutable model views verified; only safetensors paths are external links.')
