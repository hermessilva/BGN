#!/usr/bin/env python3
"""One-time migration tool: convert the pickled (ray/torch) checkpoints shipped
with Bidirectional GaitNet into plain `.safetensors` files that the C++/Eigen
inference code can read without any Python at runtime.

Run once (needs only numpy):

    python tools/export_weights.py

Produces, next to each source checkpoint, a `<name>.safetensors` holding all the
network weights as float32 plus a `__metadata__` map (env xml, type, cascading).

The ray "worker" blob inside the policy checkpoints references ray classes; we do
NOT need ray installed — a permissive Unpickler stubs every non-numpy class and we
read the numpy arrays straight out of the reconstructed object graph.
"""
import io
import json
import os
import pickle
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- permissive unpickler: stub every non-numpy class, keep numpy arrays ------
class _Stub:
    def __init__(self, *a, **k):
        self.__dict__["_args"] = a

    def __setstate__(self, st):
        self.__dict__["_state"] = st

    def __call__(self, *a, **k):
        return _Stub()

    def __setitem__(self, k, v):
        self.__dict__.setdefault("_items", {})[k] = v

    def append(self, *a):
        pass


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy") or module in (
            "_codecs", "collections", "builtins", "copyreg",
        ):
            try:
                return super().find_class(module, name)
            except Exception:
                return _Stub
        return _Stub


def _loads_stub(blob):
    return _StubUnpickler(io.BytesIO(blob)).load()


# --- minimal safetensors writer ----------------------------------------------
def write_safetensors(path, tensors, metadata):
    """tensors: dict name -> np.ndarray (cast to float32, row-major)."""
    header = {}
    blobs = []
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        data = arr.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # 8-byte little-endian header length, then header, then data
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)
    total = 8 + len(hjson) + offset
    print(f"  wrote {os.path.relpath(path, ROOT)}  ({len(tensors)} tensors, {total/1e6:.1f} MB)")


def _state_dict_to_f32(sd, prefix=""):
    out = {}
    for k, v in sd.items():
        out[prefix + k] = np.asarray(v, dtype=np.float32)
    return out


def export_fgn(src):
    s = pickle.load(open(src, "rb"))
    tensors = _state_dict_to_f32(s["ref"], "ref.")
    meta = {"type": "fgn", "env_xml": s.get("metadata") or "",
            "is_cascaded": int(bool(s.get("is_cascaded", False)))}
    write_safetensors(src + ".safetensors", tensors, meta)


def export_bgn(src):
    s = pickle.load(open(src, "rb"))
    tensors = _state_dict_to_f32(s["gvae"], "gvae.")
    meta = {"type": "bgn", "env_xml": s.get("metadata") or ""}
    write_safetensors(src + ".safetensors", tensors, meta)


def export_policy(src):
    s = pickle.load(open(src, "rb"))
    worker = _loads_stub(s["worker"])
    weights = worker["state"]["default_policy"]["weights"]
    tensors = {}
    # policy actor (p_fc.*) and log_std are all we need for inference (get_action = loc)
    for k, v in weights.items():
        if k.startswith("p_fc.") or k == "log_std":
            tensors["policy." + k] = np.asarray(v, dtype=np.float32)
    # muscle network (two-level controller)
    if "muscle" in s and s["muscle"] is not None:
        for k, v in s["muscle"].items():
            tensors["muscle." + k] = np.asarray(v, dtype=np.float32)
    meta = {"type": "policy", "env_xml": s.get("metadata") or "",
            "cascading": int(bool(s.get("cascading", False)))}
    write_safetensors(src + ".safetensors", tensors, meta)


def main():
    jobs = [
        ("fgn/fgn_narrow_model_entire_01", export_fgn),
        ("bgn/bgn_narrow_model_entire_01", export_bgn),
        ("data/trained_nn/ankle_no_mesh_lbs", export_policy),
        ("data/trained_nn/hip_no_mesh_lbs", export_policy),
        ("data/trained_nn/merge_no_mesh_lbs", export_policy),
        ("data/trained_nn/skel_no_mesh_lbs", export_policy),
    ]
    for rel, fn in jobs:
        src = os.path.join(ROOT, rel)
        if not os.path.exists(src):
            print(f"  SKIP (missing) {rel}")
            continue
        print(f"exporting {rel} ...")
        fn(src)


if __name__ == "__main__":
    main()
