#!/usr/bin/env python3
"""Reproduce RT-12 exact typed M2 fixtures with independent NumPy expectations."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import google.protobuf
import numpy as np
import onnx
from onnx import TensorProto, checker, defs, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from check_inference import VERSIONS, REVISION, LICENSE

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    actual = {"python": platform.python_version(), "onnx": onnx.__version__,
              "numpy": np.__version__, "protobuf": google.protobuf.__version__}
    if actual != VERSIONS:
        raise SystemExit("reference versions differ: " + repr(actual))
    artifacts, cases = {}, []

    def artifact(path, payload):
        artifacts[path] = payload
        return {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}

    def value(case, role, name, array):
        wire = numpy_helper.from_array(np.asarray(array), name=name)
        return {"name": name, "tensor": artifact(f"m2/values/{case}/{role}_{name}.pb",
                                                   wire.SerializeToString(deterministic=True))}

    def add(name, operator, feeds, expected, **attributes):
        node = helper.make_node(operator, list(feeds), ["y"], name=name, **attributes)
        dtype = lambda array: TensorProto.FLOAT if array.dtype == np.float32 else TensorProto.INT64
        graph = helper.make_graph([node], name,
            [helper.make_tensor_value_info(key, dtype(array), list(array.shape)) for key, array in feeds.items()],
            [helper.make_tensor_value_info("y", dtype(expected), list(expected.shape))])
        model = helper.make_model(graph, producer_name="itsmontoya/onnx RT-12", ir_version=8,
                                  opset_imports=[helper.make_opsetid("", 13)])
        checker.check_model(model, full_check=True)
        observation, status, reason = {}, "agreement", ""
        try:
            returned = ReferenceEvaluator(model).run(None, feeds)[0]
            observation = {"status": "returned", "shape": list(returned.shape)}
            if returned.shape != expected.shape:
                if operator != "Gather" or expected.size != 0:
                    raise ValueError("unexpected reference shape")
                status, reason = "difference", "Evaluator empty Gather returns [0]; exact prefix/index/suffix shape confirmed by NumPy take."
            else:
                np.testing.assert_array_equal(returned, expected)
        except ValueError as error:
            if operator != "Flatten" or expected.size != 0:
                raise
            status, reason = "extension", "Evaluator Flatten infers -1 through an empty prefix; explicit checked products remain defined."
            observation = {"status": "rejected", "error": str(error)}
        cases.append({"name": name, "origin": "tools/check_m2.py; original graph and independent NumPy expectation",
                      "license": LICENSE, "status": status, "reason": reason, "ir_version": 8, "opsets": {"": 13},
                      "schemas": [{"domain": "", "operator": operator, "version": defs.get_schema(operator, 13).since_version}],
                      "model": artifact(f"m2/models/{name}.onnx", model.SerializeToString(deterministic=True)),
                      "comparison": {"absolute": 0, "relative": 0, "signed_zero": True},
                      "inputs": [value(name, "input", key, array) for key, array in feeds.items()],
                      "outputs": [value(name, "output", "y", expected)],
                      "observations": {"reference": observation, "independent": "NumPy concrete shape, reshape, transpose, squeeze, expand_dims, take or concatenate"}})

    i = lambda values: np.asarray(values, dtype=np.int64)
    for dtype in [np.float32, np.int64]:
        suffix = np.dtype(dtype).name
        x = np.arange(1, 25, dtype=dtype).reshape(2, 3, 4)
        if dtype == np.float32:
            x.reshape(-1)[:5] = np.asarray([0x80000000, 0, 0x7f800000, 0xff800000, 0x7fc00001], np.uint32).view(np.float32)
        if dtype == np.int64:
            x.flat[0], x.flat[1], x.flat[2] = 2**63-1, -2**63, 2**53+1
        for axis in [0, 1, -1, 3]:
            split = axis if axis >= 0 else x.ndim + axis
            expected = x.reshape(int(np.prod(x.shape[:split])), int(np.prod(x.shape[split:])))
            add(f"flatten_{suffix}_{axis}", "Flatten", {"x": x}, expected, axis=axis)
        add(f"flatten_default_{suffix}", "Flatten", {"x": x}, x.reshape(2, 12))
        add(f"flatten_scalar_{suffix}", "Flatten", {"x": np.asarray(7, dtype=dtype)}, np.asarray([[7]], dtype=dtype), axis=0)
        empty = np.empty((0, 2, 3), dtype=dtype)
        add(f"flatten_empty_{suffix}", "Flatten", {"x": empty}, empty.reshape(0, 6))
        add(f"transpose_default_{suffix}", "Transpose", {"x": x}, np.transpose(x))
        add(f"transpose_explicit_{suffix}", "Transpose", {"x": x}, np.transpose(x, (1, 2, 0)), perm=[1, 2, 0])
        add(f"transpose_scalar_{suffix}", "Transpose", {"x": np.asarray(7, dtype=dtype)}, np.asarray(7, dtype=dtype))
        add(f"transpose_empty_{suffix}", "Transpose", {"x": empty}, np.transpose(empty))
        add(f"reshape_zero_infer_{suffix}", "Reshape", {"x": x, "shape": i([0, -1])}, x.reshape(2, 12))
        add(f"reshape_scalar_{suffix}", "Reshape", {"x": np.asarray([7], dtype=dtype), "shape": i([])}, np.asarray(7, dtype=dtype))
        add(f"reshape_empty_{suffix}", "Reshape", {"x": empty, "shape": i([0, 6])}, empty.reshape(0, 6))
        add(f"reshape_inferred_zero_{suffix}", "Reshape", {"x": empty, "shape": i([-1, 6])}, empty.reshape(0, 6))
        unit = x.reshape(1, 2, 1, 3, 4, 1)
        add(f"squeeze_omitted_{suffix}", "Squeeze", {"x": unit}, np.squeeze(unit))
        add(f"squeeze_negative_{suffix}", "Squeeze", {"x": unit, "axes": i([-1, 0])}, np.squeeze(unit, (-1, 0)))
        add(f"squeeze_empty_axes_{suffix}", "Squeeze", {"x": unit, "axes": i([])}, unit.copy())
        add(f"squeeze_scalar_{suffix}", "Squeeze", {"x": np.asarray(7, dtype=dtype)}, np.asarray(7, dtype=dtype))
        add(f"unsqueeze_output_rank_{suffix}", "Unsqueeze", {"x": x, "axes": i([-1, 0])}, np.expand_dims(x, (-1, 0)))
        add(f"unsqueeze_scalar_{suffix}", "Unsqueeze", {"x": np.asarray(7, dtype=dtype), "axes": i([-1])}, np.asarray([7], dtype=dtype))
        add(f"unsqueeze_empty_axes_{suffix}", "Unsqueeze", {"x": empty, "axes": i([])}, empty.copy())
        for label, indices, axis in [("scalar", i(-1), 1), ("general", i([[0, -1], [1, 0]]), -2),
                                     ("empty", np.empty((0, 2), np.int64), 1)]:
            add(f"gather_{label}_{suffix}", "Gather", {"x": x, "indices": indices}, np.take(x, indices, axis=axis), axis=axis)
        for label, data in [("prefix", np.empty((0, 3, 4), dtype=dtype)),
                            ("suffix", np.empty((2, 3, 0), dtype=dtype))]:
            indices = i([[0, -1], [1, 0]])
            add(f"gather_empty_{label}_{suffix}", "Gather", {"x": data, "indices": indices},
                np.take(data, indices, axis=1), axis=1)
        zero = np.empty((2, 0, 3), dtype=dtype)
        add(f"gather_zero_axis_{suffix}", "Gather", {"x": zero, "indices": i([])}, np.take(zero, i([]), axis=1), axis=1)
        add(f"concat_negative_{suffix}", "Concat", {"x": x, "other": x}, np.concatenate((x, x), axis=-1), axis=-1)
        add(f"concat_empty_{suffix}", "Concat", {"x": empty, "other": empty}, np.concatenate((empty, empty), axis=0), axis=0)
        add(f"shape_{suffix}", "Shape", {"x": x}, i(x.shape))
        add(f"shape_scalar_{suffix}", "Shape", {"x": np.asarray(7, dtype=dtype)}, i([]))
        add(f"shape_empty_{suffix}", "Shape", {"x": empty}, i(empty.shape))
    manifest = {"format_version": 1, "onnx_source_revision": REVISION, "reference_versions": VERSIONS,
                "original_license": LICENSE, "cases": cases}
    artifacts["m2/provenance.json"] = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    for path, payload in artifacts.items():
        destination = ROOT / "testdata" / path
        if args.write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        if not destination.exists() or destination.read_bytes() != payload:
            raise SystemExit("M2 fixture differs: " + path)
    existing = {str(path.relative_to(ROOT / "testdata")) for path in (ROOT / "testdata/m2").rglob("*")
                if path.is_file() and path.suffix != ".md"}
    if existing != set(artifacts):
        raise SystemExit("unexpected or missing M2 artifacts: " + repr(existing ^ set(artifacts)))
    from collections import Counter
    print("PASS: M2 exact typed cases", dict(Counter(case["status"] for case in cases)))


if __name__ == "__main__":
    main()
