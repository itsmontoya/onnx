#!/usr/bin/env python3
"""Reproduce original RT-09 M1 graphs and pinned reference expectations."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import warnings

import google.protobuf
import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {"python": "3.9.6", "onnx": "1.16.2", "numpy": "1.26.4", "protobuf": "4.25.3"}


def record(name, array):
    array = np.asarray(array)
    result = {"name": name, "shape": list(array.shape), "dtype": str(array.dtype)}
    result["bits" if array.dtype == np.float32 else "integers"] = (
        array.reshape(-1).view(np.uint32).tolist() if array.dtype == np.float32 else array.reshape(-1).tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    actual = {"python": platform.python_version(), "onnx": onnx.__version__,
              "numpy": np.__version__, "protobuf": google.protobuf.__version__}
    if actual != VERSIONS:
        raise SystemExit("reference versions differ: " + repr(actual))
    artifacts, cases = {}, []

    def add(name, nodes, feeds, initializers=None, extension_shape=None):
        inputs = [helper.make_tensor_value_info(key, TensorProto.FLOAT, list(value.shape)) for key, value in feeds.items()]
        output_type = TensorProto.INT64 if name.startswith("constant_int") else TensorProto.FLOAT
        graph = helper.make_graph(nodes, name, inputs, [helper.make_tensor_value_info("y", output_type, None)])
        if initializers:
            graph.initializer.extend(numpy_helper.from_array(value, name=key) for key, value in initializers.items())
        model = helper.make_model(graph, producer_name="itsmontoya/onnx RT-09", opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        # Infer a real output declaration before checker validation.
        model = onnx.shape_inference.infer_shapes(model)
        checker.check_model(model, full_check=True)
        status, reference_error = "agreement", None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                values = ReferenceEvaluator(model).run(None, feeds)[0]
            except ValueError as error:
                if extension_shape is None:
                    raise
                status, reference_error = "extension_reference_rejected", str(error)
                values = np.empty(extension_shape, dtype=np.float32)
        payload = model.SerializeToString(deterministic=True)
        artifacts[name + ".onnx"] = payload
        cases.append({"name": name, "model": name + ".onnx", "sha256": hashlib.sha256(payload).hexdigest(),
                      "reference_status": status, "reference_error": reference_error,
                      "inputs": [record(key, value) for key, value in feeds.items()], "outputs": [record("y", values)]})

    f = lambda values: np.asarray(values, dtype=np.float32)
    single = lambda op, inputs, **attrs: [helper.make_node(op, inputs, ["y"], name=op.lower(), **attrs)]
    for op in ["Add", "Sub", "Mul", "Div"]:
        add(op.lower() + "_matching", single(op, ["a", "b"]), {"a": f([1, 2, 3]), "b": f([2, 4, 8])})
        add(op.lower() + "_broadcast", single(op, ["a", "b"]), {"a": f([[1], [3]]), "b": f([[2, 4, 8]])})
        add(op.lower() + "_scalar", single(op, ["a", "b"]), {"a": f(2), "b": f([1, 2, 4])})
        add(op.lower() + "_empty", single(op, ["a", "b"]), {"a": np.empty((2, 0, 3), np.float32), "b": f(1)})
    add("div_nonfinite", single("Div", ["a", "b"]), {"a": f([1, -1, 0, np.inf, 1]), "b": f([0, -0.0, 0, np.inf, -np.inf])})
    for name, a, b in [
        ("dot", f([1, 2]), f([3, 4])), ("vector_matrix", f([1, 2]), f([[1, 2, 3], [4, 5, 6]])),
        ("matrix_vector", f([[1, 2], [3, 4]]), f([2, 3])),
        ("batch", np.arange(12, dtype=np.float32).reshape(2, 1, 2, 3), np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2)),
        ("empty_contraction", np.empty((2, 0), np.float32), np.empty((0, 3), np.float32)),
        ("empty_output", np.empty((0, 2), np.float32), f([[1], [2]]))]:
        add("matmul_" + name, single("MatMul", ["a", "b"]), {"a": a, "b": b})
    for ta in [0, 1]:
        for tb in [0, 1]:
            for bias_name, bias in [("absent", None), ("scalar", f(2)), ("vector", f([1, 2, 3])),
                                    ("row", f([[1, 2, 3]])), ("column", f([[1], [2]])), ("full", f([[1, 2, 3], [4, 5, 6]]))]:
                a, b = f([[1, 2], [3, 4]]), f([[1, 2, 3], [4, 5, 6]])
                feeds = {"a": a.T.copy() if ta else a, "b": b.T.copy() if tb else b}
                if bias is not None:
                    feeds["c"] = bias
                add(f"gemm_{ta}{tb}_{bias_name}", single("Gemm", list(feeds), transA=ta, transB=tb, alpha=0.5, beta=2.0), feeds)
    add("gemm_defaults", single("Gemm", ["a", "b", ""]), {"a": f([[2]]), "b": f([[3]])})
    add("gemm_beta_zero", single("Gemm", ["a", "b", "c"], beta=0.0), {"a": f([[2]]), "b": f([[3]]), "c": f(np.nan)})
    add("gemm_empty_contraction", single("Gemm", ["a", "b", "c"]), {"a": np.empty((2, 0), np.float32), "b": np.empty((0, 3), np.float32), "c": f(2)})
    for op in ["Relu", "Sigmoid", "Tanh"]:
        add(op.lower() + "_finite", single(op, ["x"]), {"x": f([-2, -0.0, 0, 1, 3])})
        add(op.lower() + "_nonfinite", single(op, ["x"]), {"x": f([-np.inf, np.inf, np.nan])})
        add(op.lower() + "_scalar", single(op, ["x"]), {"x": f(-1)})
        add(op.lower() + "_empty", single(op, ["x"]), {"x": np.empty((0, 3), np.float32)})
    for axis in [-1, 0, 1, -2]:
        add("softmax_axis_" + str(axis), single("Softmax", ["x"], axis=axis), {"x": np.arange(12, dtype=np.float32).reshape(2, 3, 2)})
    add("softmax_default", single("Softmax", ["x"]), {"x": f([[1, 2, 3], [4, 5, 6]])})
    add("softmax_nonfinite", single("Softmax", ["x"]), {"x": f([[np.inf, 0], [-np.inf, -np.inf], [np.nan, 1], [-np.inf, 0]])})
    add("softmax_empty_axis", single("Softmax", ["x"], axis=1), {"x": np.empty((2, 0, 3), np.float32)}, extension_shape=(2, 0, 3))
    add("identity", single("Identity", ["x"]), {"x": f([[2, -3]])})
    for suffix, attr in [
        ("float", helper.make_attribute("value_float", 1.5)), ("floats", helper.make_attribute("value_floats", [1.0, -2.0])),
        ("int", helper.make_attribute("value_int", 9223372036854775807)),
        ("ints", helper.make_attribute("value_ints", [16777217, -9223372036854775808])),
        ("floats_empty", onnx.AttributeProto(name="value_floats", type=onnx.AttributeProto.FLOATS)),
        ("ints_empty", onnx.AttributeProto(name="value_ints", type=onnx.AttributeProto.INTS)),
        ("tensor", helper.make_attribute("value", helper.make_tensor("", TensorProto.FLOAT, [2], [3, 4]))),
        ("int_tensor", helper.make_attribute("value", helper.make_tensor("", TensorProto.INT64, [2], [16777217, 9223372036854775807])))]:
        node = helper.make_node("Constant", [], ["y"], name="constant")
        node.attribute.append(attr)
        name = "constant_" + suffix
        if suffix == "int_tensor": name = "constant_int_tensor"
        add(name, [node], {})
    feeds = {"x": f([[1, 2], [3, 4]]), "w": f([[1, -1], [2, 1]]), "b": f([1, -2])}
    add("mlp", [helper.make_node("Gemm", ["x", "w", "b"], ["h"]), helper.make_node("Relu", ["h"], ["r"]), helper.make_node("Softmax", ["r"], ["y"])], feeds)
    add("residual", [helper.make_node("MatMul", ["x", "w"], ["h"]), helper.make_node("Add", ["h", "b"], ["r"]), helper.make_node("Relu", ["r"], ["z"]), helper.make_node("Add", ["z", "x"], ["y"])], feeds)
    mlp_nodes = [helper.make_node("Gemm", ["x", "w", "b"], ["h"]), helper.make_node("Relu", ["h"], ["r"]), helper.make_node("Softmax", ["r"], ["y"])]
    add("mlp_initializers", mlp_nodes, {"x": feeds["x"]}, initializers={"w": feeds["w"], "b": feeds["b"]})
    manifest = {"origin": "tools/check_m1.py; original repository graphs and values", "license": "Original repository fixtures; no upstream model copied",
                "onnx_source_revision": "3bf92c03a9f27eba3bda1e5b9e63ea20ec213557", "reference_versions": VERSIONS,
                "ir_version": 8, "opsets": {"": 13}, "comparison": "finite atol=1e-5 rtol=1e-5; NaN classification; infinity signs; signed zero for elementwise/empty contraction cases",
                "cases": cases}
    artifacts["provenance.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    for name, payload in artifacts.items():
        path = ROOT / "testdata/m1" / name
        if args.write:
            path.write_bytes(payload)
        if not path.exists() or path.read_bytes() != payload:
            raise SystemExit("M1 fixture differs: " + name)
    print(f"PASS: {len(cases)} checker-validated M1 models; {sum(c['reference_status'] == 'agreement' for c in cases)} reference agreements; empty-axis extension retained")


if __name__ == "__main__":
    main()
