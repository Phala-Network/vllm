"""Compare fresh per-node startup log captures, requiring all TP8 workers."""
import argparse
import hashlib
import json
from pathlib import Path

MARKER = 'PHALA_KV_LAYOUT_AUDIT '

def read(path, dcp_size):
    records = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if MARKER not in line:
            continue
        row = json.loads(line.split(MARKER, 1)[1])
        assert row['schema'] == 'phala.mooncake.layout.v1'
        canonical = json.dumps(row['compatibility'], sort_keys=True, separators=(',', ':'))
        assert hashlib.sha256(canonical.encode()).hexdigest() == row['compatibility_sha256']
        rank = tuple(sorted(row['compatibility']['ranks'].items()))
        if rank in records:
            assert records[rank] == row, 'Conflicting repeated rank records; capture only the current startup'
        records[rank] = row
    assert len(records) == 8, f'Expected eight worker records, found {len(records)}'
    assert {dict(key)['tp_rank'] for key in records} == set(range(8))
    assert all(dict(key)['tp_size'] == 8 and dict(key)['dcp_size'] == dcp_size for key in records)
    return records

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('node101')
    ap.add_argument('node102')
    ap.add_argument('--dcp-size', type=int, choices=[1, 8], required=True)
    args=ap.parse_args()
    a,b=read(args.node101,args.dcp_size),read(args.node102,args.dcp_size)
    assert a.keys() == b.keys(), 'Rank topology differs'
    mismatches=[dict(k) for k in a if a[k]['compatibility'] != b[k]['compatibility']]
    output={'status':'FAIL' if mismatches else 'PASS','ranks_compared':len(a),'mismatched_ranks':mismatches,'capacity_equal_by_tp_rank':{dict(k)['tp_rank']:a[k]['capacity']==b[k]['capacity'] for k in a},'scope':'actual dtype, grouped shape/strides, full layer/component mapping, storage alias/offset and db transfer segments; not a functional Store hit test'}
    print(json.dumps(output,indent=2))
    raise SystemExit(bool(mismatches))

if __name__=='__main__':
    main()
