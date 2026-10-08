"""One registration-time metadata record; never logs addresses or tensor values."""
import dataclasses
import enum
import hashlib
import json


def _plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return _plain(value.value)
    if dataclasses.is_dataclass(value):
        return {"spec_type": type(value).__name__, **{f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        return str(value)
    raise TypeError(f"Unsupported layout metadata type: {type(value).__name__}")


def make_layout_record(worker, kv_caches, group_kernel_blocks):
    # Addresses are used only as ephemeral identity keys, never serialized.
    storages = {}
    storage_ranges = []
    capacity = {"num_blocks": worker.num_blocks, "groups": [], "storages": []}
    compatibility = {
        "ranks": {name: getattr(worker, name) for name in ("tp_rank", "tp_size", "pp_rank", "pp_size", "pcp_rank", "pcp_size", "dcp_rank", "dcp_size")},
        "block_size": worker.block_size,
        "hash_block_size": worker.hash_block_size,
        "cache_layout": str(worker.cache_config.get_resolved_kv_cache_layout()),
        "groups": [],
        "databases": [],
    }
    for group in worker._kv_cache_groups:
        blocks = worker._kv_cache_config.hisparse_host_num_blocks if group.host_resident else worker.num_blocks
        if not blocks:
            raise ValueError("Layout audit requires a positive block count")
        group_out = {"spec": _plain(group.kv_cache_spec), "host_resident": group.host_resident, "layer_names": list(group.layer_names), "layers": []}
        cap_group = {"num_blocks": blocks, "layers": []}
        for name in group.layer_names:
            value = kv_caches[name]
            components = value if isinstance(value, list) else [value]
            layer = {"name": name, "components": []}
            cap_layer = {"name": name, "components": []}
            for component in components:
                tensor = group_kernel_blocks(component, blocks)
                storage = tensor.untyped_storage()
                storage_key = (str(tensor.device), storage.data_ptr())
                if storage_key not in storages:
                    alias = len(storages)
                    storages[storage_key] = alias
                    storage_ranges.append((str(tensor.device), storage.data_ptr(), storage.nbytes(), alias))
                    capacity["storages"].append({"alias": alias, "nbytes": storage.nbytes()})
                alias = storages[storage_key]
                element_size = tensor.element_size()
                layer["components"].append({
                    "dtype": str(tensor.dtype), "device_type": tensor.device.type,
                    "element_size": element_size, "shape_per_block": list(tensor.shape[1:]),
                    "stride_bytes": [s * element_size for s in tensor.stride()],
                    "storage_alias": alias, "storage_offset_bytes": tensor.storage_offset() * element_size,
                    "storage_bytes_per_block": [storage.nbytes(), blocks] if storage.nbytes() % blocks else storage.nbytes() // blocks,
                })
                cap_layer["components"].append({"raw_shape": list(component.shape), "grouped_shape": list(tensor.shape)})
            group_out["layers"].append(layer)
            cap_group["layers"].append(cap_layer)
        compatibility["groups"].append(group_out)
        capacity["groups"].append(cap_group)
    for db in worker.token_dbs:
        layout = db.store_layout
        # Current K3 TP8/DCP8 selects RankLocalStoreLayout. Do not manufacture
        # equivalent evidence for a layout with a different transfer contract.
        if not hasattr(layout, "block_len") or not hasattr(layout, "kv_caches_base_addr"):
            raise TypeError(f"Unqualified Store layout for audit: {type(layout).__name__}")
        segments = []
        for address, length in zip(layout.kv_caches_base_addr, layout.block_len, strict=True):
            matches = [(alias, address - base) for device, base, size, alias in storage_ranges if base <= address < base + size]
            if len(matches) != 1:
                raise ValueError("Store segment does not have one known tensor storage")
            alias, offset = matches[0]
            segments.append({"storage_alias": alias, "storage_offset_bytes": offset, "block_len": length})
        compatibility["databases"].append({"layout_type": type(layout).__name__, "block_len": list(layout.block_len), "segments": segments})
    canonical = json.dumps(compatibility, sort_keys=True, separators=(",", ":"))
    return {"schema": "phala.mooncake.layout.v1", "compatibility_sha256": hashlib.sha256(canonical.encode()).hexdigest(), "compatibility": compatibility, "capacity": capacity}


def log_layout(worker, kv_caches, group_kernel_blocks, logger):
    logger.info("PHALA_KV_LAYOUT_AUDIT %s", json.dumps(make_layout_record(worker, kv_caches, group_kernel_blocks), sort_keys=True, separators=(",", ":")))
