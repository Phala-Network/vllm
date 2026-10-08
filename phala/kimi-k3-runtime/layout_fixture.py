"""CPU fixture: capacity independence, real stride sensitivity, all state parts."""
import dataclasses
import json
from types import SimpleNamespace as NS
import torch
from layout_audit import make_layout_record

@dataclasses.dataclass
class StateSpec:
    block_size: int = 16
    dtype: object = torch.float32

def fixture(blocks, transposed=False):
    raw = torch.empty((blocks, 24), dtype=torch.float32)
    a = raw[:, :8].view(blocks, 2, 4)
    if transposed:
        # Same shape, genuine changed stride without reading tensor contents.
        a = raw[:, :8].view(blocks, 4, 2).transpose(1, 2)
    b = raw[:, 8:24].view(blocks, 4, 4)
    layout = NS(block_len=[96], kv_caches_base_addr=[raw.data_ptr()])
    worker = NS(**{key: 0 if key.endswith('_rank') else 8 for key in ('tp_rank','tp_size','pp_rank','pp_size','pcp_rank','pcp_size','dcp_rank','dcp_size')}, num_blocks=blocks, block_size=16, hash_block_size=16, cache_config=NS(get_resolved_kv_cache_layout=lambda:'LBHNC'), _kv_cache_config=NS(hisparse_host_num_blocks=None), _kv_cache_groups=[NS(host_resident=False, layer_names=['kda.0'], kv_cache_spec=StateSpec())], token_dbs=[NS(store_layout=layout)])
    return make_layout_record(worker, {'kda.0':[a,b]}, lambda t,n:t)

a,b,c = fixture(3),fixture(7),fixture(3,True)
assert a['compatibility_sha256'] == b['compatibility_sha256']
assert a['capacity'] != b['capacity']
assert a['compatibility_sha256'] != c['compatibility_sha256']
assert len(a['compatibility']['groups'][0]['layers'][0]['components']) == 2
encoded=json.dumps(a)
assert 'data_ptr' not in encoded and 'address' not in encoded
assert json.loads(encoded)==a
print(json.dumps({'status':'PASS','capacity_change_preserves_compatibility':True,'actual_stride_change_detected':True,'all_state_components':2,'json_roundtrip':True}))
