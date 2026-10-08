"""CPU-only: tokenizer + actual HF image processor, with no weight mounts."""
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer
from vllm.transformers_utils.config import get_config

root = '/opt/runtime-model/target'
target_config = get_config(root, trust_remote_code=True)
draft_config = get_config('/opt/runtime-model/draft', trust_remote_code=True)
assert type(target_config).__module__.startswith('vllm.')
assert type(draft_config).__module__.startswith('vllm.')
tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True, local_files_only=True)
processor = AutoImageProcessor.from_pretrained(root, trust_remote_code=True, local_files_only=True)
tokens = tokenizer.encode('Hello world')
assert tokens and tokenizer.decode(tokens)
output = processor.preprocess([{'type': 'image', 'image': Image.new('RGB', (56, 56), 'white')}], return_tensors='pt')
assert output
manifest = json.loads(Path('/opt/runtime-model/manifest.json').read_text())
expected = {f['name']: f['sha256'] for f in manifest['models']['target']['files'] if f['name'].endswith('.py')}
loaded = {}
for cls in (type(tokenizer), type(processor)):
    path = Path(inspect.getfile(cls))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[path.name]
    loaded[cls.__name__] = str(path)
subprocess.run(['hf', '--help'], check=True, stdout=subprocess.DEVNULL)
subprocess.run(['hf', 'download', '--help'], check=True, stdout=subprocess.DEVNULL)
subprocess.run(['hf', 'cache', 'verify', '--help'], check=True, stdout=subprocess.DEVNULL)
print(json.dumps({'target_config': type(target_config).__module__, 'draft_config': type(draft_config).__module__, 'tokenizer_tokens': len(tokens), 'processor_keys': list(output.keys()), 'loaded_code': loaded, 'hf_cli': 'help, download and cache verify commands available', 'weights_mounted': False}))
