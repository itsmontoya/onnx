#!/usr/bin/env python3
"""Reproduce RT-13 CNN fixtures, independent formulas and both pinned oracles."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import warnings

import google.protobuf
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, defs, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from check_conformance import observe, values_record, VERSIONS
from check_inference import REVISION, LICENSE

ROOT = Path(__file__).resolve().parents[1]


def spatial(size, kernel, stride, dilation, begin, end, ceil):
    effective = (kernel - 1) * dilation + 1
    numerator = size + begin + end - effective
    result = max(0, (numerator + (stride - 1 if ceil else 0)) // stride + 1)
    if ceil:
        result = min(result, (size + begin + stride - 1) // stride)
    return result


def geometry(x, kernel, attributes):
    strides, dilations = attributes.get("strides", [1, 1]), attributes.get("dilations", [1, 1])
    pads = list(attributes.get("pads", [0, 0, 0, 0]))
    auto = attributes.get("auto_pad", "NOTSET")
    if auto in ["SAME_UPPER", "SAME_LOWER"]:
        for axis, size in enumerate(x.shape[2:]):
            output = (size + strides[axis] - 1) // strides[axis]
            total = max(0, (output - 1) * strides[axis] + (kernel[axis] - 1) * dilations[axis] + 1 - size) if size else 0
            pads[axis], pads[axis+2] = total // 2, total - total // 2
            if auto == "SAME_LOWER":
                pads[axis], pads[axis+2] = pads[axis+2], pads[axis]
    shape = [spatial(x.shape[axis+2], kernel[axis], strides[axis], dilations[axis],
                     pads[axis], pads[axis+2], attributes.get("ceil_mode", 0)) for axis in range(2)]
    return strides, dilations, pads, shape


def convolution(x, w, b=None, **attributes):
    strides, dilations, pads, spatial_shape = geometry(x, w.shape[2:], attributes)
    groups = attributes.get("group", 1)
    output = np.zeros((x.shape[0], w.shape[0], *spatial_shape), np.float32)
    padded = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    for row in range(spatial_shape[0]):
        for col in range(spatial_shape[1]):
            window = padded[:, :, row*strides[0]:row*strides[0]+(w.shape[2]-1)*dilations[0]+1:dilations[0],
                            col*strides[1]:col*strides[1]+(w.shape[3]-1)*dilations[1]+1:dilations[1]]
            for group in range(groups):
                channels = x.shape[1] // groups
                filters = w.shape[0] // groups
                output[:, group*filters:(group+1)*filters, row, col] = np.einsum(
                    "nchw,mchw->nm", window[:, group*channels:(group+1)*channels],
                    w[group*filters:(group+1)*filters], optimize=False)
    if b is not None:
        output += b.reshape(1, -1, 1, 1)
    return output


def pooling(operation, x, **attributes):
    if operation == "GlobalAveragePool":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.mean(x, axis=(2, 3), keepdims=True, dtype=np.float32)
    kernel = attributes["kernel_shape"]
    strides, dilations, pads, spatial_shape = geometry(x, kernel, attributes)
    output = np.empty((*x.shape[:2], *spatial_shape), np.float32)
    for row in range(spatial_shape[0]):
        for col in range(spatial_shape[1]):
            rows = np.arange(kernel[0]) * dilations[0] + row * strides[0] - pads[0]
            cols = np.arange(kernel[1]) * dilations[1] + col * strides[1] - pads[1]
            rows, cols = rows[(rows >= 0) & (rows < x.shape[2])], cols[(cols >= 0) & (cols < x.shape[3])]
            window = x[:, :, rows[:, None], cols]
            if operation == "MaxPool":
                output[:, :, row, col] = np.max(window, axis=(2, 3)) if rows.size * cols.size else -np.inf
            else:
                divisor = np.prod(kernel) if attributes.get("count_include_pad", 0) else rows.size * cols.size
                with np.errstate(all="ignore"):
                    output[:, :, row, col] = np.sum(window, axis=(2, 3), dtype=np.float32) / np.float32(divisor)
    return output


def normalization(feeds, epsilon=1e-5, **ignored):
    x, scale, bias, mean, variance = feeds.values()
    shape = (1, x.shape[1]) + (1,) * (x.ndim-2)
    with np.errstate(all="ignore"):
        return (scale.reshape(shape) * (x - mean.reshape(shape)) /
                np.sqrt(variance.reshape(shape) + np.float32(epsilon)) + bias.reshape(shape))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    actual = {"python": platform.python_version(), "onnx": onnx.__version__, "numpy": np.__version__,
              "protobuf": google.protobuf.__version__, "onnxruntime": ort.__version__}
    if actual != VERSIONS:
        raise SystemExit("reference versions differ: " + repr(actual))
    artifacts, cases = {}, []

    def artifact(path, payload):
        artifacts[path] = payload
        return {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}

    def value(case, role, name, array):
        wire = numpy_helper.from_array(np.asarray(array, dtype=np.float32), name=name)
        return {"name": name, "tensor": artifact(f"m3/values/{case}/{role}_{name}.pb", wire.SerializeToString(deterministic=True))}

    def agrees(observation, expected):
        if observation["status"] != "returned" or len(observation["outputs"]) != len(expected):
            return False
        for observed, wanted in zip(observation["outputs"], expected.values()):
            returned = np.asarray(observed["bits"], np.uint32).view(np.float32).reshape(observed["shape"])
            if returned.shape != wanted.shape or not np.allclose(returned, wanted, atol=1e-5, rtol=1e-5, equal_nan=True):
                return False
        return True

    def add(name, operation, feeds, expected=None, empty_output=False, model=None, **attributes):
        if expected is None:
            if operation == "Conv":
                expected = convolution(*feeds.values(), **attributes)
            elif operation == "BatchNormalization":
                expected = normalization(feeds, **attributes)
            else:
                expected = pooling(operation, feeds["x"], **attributes)
        expected = {"y": expected} if isinstance(expected, np.ndarray) else expected
        if model is None:
            node = helper.make_node(operation, list(feeds), ["y", ""] if empty_output else ["y"], name=name, **attributes)
            graph = helper.make_graph([node], name,
                [helper.make_tensor_value_info(key, TensorProto.FLOAT, list(array.shape)) for key, array in feeds.items()],
                [helper.make_tensor_value_info(key, TensorProto.FLOAT, list(array.shape)) for key, array in expected.items()])
            model = helper.make_model(graph, producer_name="itsmontoya/onnx RT-13", ir_version=8,
                                      opset_imports=[helper.make_opsetid("", 13)])
        checker.check_model(model)
        observations = {}
        for oracle, use_ort in [("reference", False), ("ort", True)]:
            try:
                observations[oracle] = observe(model, feeds, use_ort=use_ort)
            except (IndexError, AssertionError, TypeError, ZeroDivisionError,
                    ort.capi.onnxruntime_pybind11_state.InvalidArgument,
                    ort.capi.onnxruntime_pybind11_state.Fail,
                    ort.capi.onnxruntime_pybind11_state.RuntimeException) as error:
                observations[oracle] = {"status": "rejected", "error": str(error)}
            observations[oracle]["matches_independent"] = agrees(observations[oracle], expected)
        status = "agreement" if observations["reference"]["matches_independent"] else (
            "extension" if observations["reference"]["status"] == "rejected" else "difference")
        reasons = {
            "Conv": "Independent NCHW grouped cross-correlation with checked SAME/VALID/floor geometry; actual evaluator automatic-padding shape/values or empty-geometry rejection retained.",
            "BatchNormalization": "Independent stored-statistics float32 formula; schema-9 evaluator applies momentum. ORT zero variance/epsilon may also differ.",
            "CNN": "Independent Conv/window/stored-statistics composition matches ORT; actual evaluator differs, including schema-9 momentum behavior."
        }
        reason = "" if status == "agreement" else reasons.get(operation,
            "Independent checked SAME/ceil/window reduction; actual evaluator shape/values or rejection retained for NaN, empty or all-padding behavior.")
        nodes = model.graph.node
        cases.append({"name": name, "origin": "tools/check_m3.py; original graph and independent NumPy formula",
                      "license": LICENSE, "status": status, "reason": reason, "ir_version": 8, "opsets": {"": 13},
                      "schemas": [{"domain": "", "operator": node.op_type,
                                   "version": defs.get_schema(node.op_type, 13).since_version} for node in nodes],
                      "model": artifact("m3/models/" + ("cnn" if operation == "CNN" else name) + ".onnx", model.SerializeToString(deterministic=True)),
                      "comparison": {"absolute": 1e-5, "relative": 1e-5, "signed_zero": False},
                      "inputs": [value(name, "input", key, array) for key, array in feeds.items()],
                      "outputs": [value(name, "output", key, array) for key, array in expected.items()],
                      "observations": dict(observations, independent="NumPy padded-window/einsum, indexed-window reductions or literal stored-statistics formula")})

    f = lambda values: np.asarray(values, np.float32)
    data = lambda shape: (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 13 - 6) / 4
    for label, xshape, wshape, attrs in [
        ("default", (2, 2, 4, 5), (3, 2, 2, 3), {}),
        ("grouped", (2, 4, 5, 6), (6, 2, 2, 3), {"group": 2, "pads": [1, 0, 0, 2]}),
        ("depthwise", (2, 3, 4, 5), (6, 1, 3, 2), {"group": 3, "strides": [2, 1]}),
        ("dilated", (1, 2, 5, 6), (4, 2, 2, 3), {"dilations": [2, 2], "strides": [1, 2], "pads": [1, 2, 0, 1]}),
        ("same_upper", (2, 2, 4, 5), (3, 2, 3, 2), {"auto_pad": "SAME_UPPER", "strides": [2, 2]}),
        ("same_lower", (2, 2, 4, 5), (3, 2, 3, 2), {"auto_pad": "SAME_LOWER", "strides": [2, 2]}),
        ("same_dilated", (1, 2, 4, 5), (3, 2, 2, 2), {"auto_pad": "SAME_LOWER", "dilations": [2, 3], "strides": [2, 2]}),
        ("valid", (1, 2, 4, 5), (3, 2, 2, 2), {"auto_pad": "VALID", "kernel_shape": [2, 2]}),
        ("empty_batch", (0, 2, 4, 5), (3, 2, 2, 2), {}),
        ("oversized", (1, 2, 1, 1), (3, 2, 3, 3), {}),
        ("padded_empty", (1, 2, 0, 2), (3, 2, 2, 2), {"pads": [2, 1, 2, 0]})]:
        for bias in [False, True]:
            feeds = {"x": data(xshape), "w": data(wshape)}
            if bias:
                feeds["b"] = data((wshape[0],))
            add(f"conv_{label}_{'bias' if bias else 'no_bias'}", "Conv", feeds, **attrs)
    for operation in ["MaxPool", "AveragePool"]:
        prefix = "max" if operation == "MaxPool" else "average"
        for label, shape, attrs in [
            ("default", (2, 2, 4, 5), {"kernel_shape": [2, 3]}),
            ("negative_pads", (1, 1, 3, 3), {"kernel_shape": [2, 2], "pads": [1, 0, 0, 1]}),
            ("ceil_partial", (1, 1, 3, 3), {"kernel_shape": [2, 2], "strides": [2, 2], "ceil_mode": 1}),
            ("ceil_trim", (1, 1, 3, 3), {"kernel_shape": [3, 3], "strides": [3, 3], "pads": [0, 0, 2, 2], "ceil_mode": 1}),
            ("same_ceil", (1, 1, 5, 4), {"kernel_shape": [2, 3], "strides": [3, 2], "auto_pad": "SAME_LOWER", "ceil_mode": 1}),
            ("ceil_negative_numerator", (1, 1, 1, 1), {"kernel_shape": [2, 2], "strides": [2, 2], "ceil_mode": 1}),
            ("same_upper", (2, 2, 4, 5), {"kernel_shape": [3, 2], "strides": [2, 2], "auto_pad": "SAME_UPPER"}),
            ("same_lower", (2, 2, 4, 5), {"kernel_shape": [3, 2], "strides": [2, 2], "auto_pad": "SAME_LOWER"}),
            ("valid", (1, 1, 4, 5), {"kernel_shape": [2, 2], "strides": [2, 2], "auto_pad": "VALID"}),
            ("all_padding", (1, 1, 1, 1), {"kernel_shape": [2, 2], "pads": [2, 2, 0, 0]}),
            ("empty_batch", (0, 2, 3, 3), {"kernel_shape": [2, 2]}),
            ("empty_spatial", (1, 1, 0, 2), {"kernel_shape": [2, 2], "pads": [2, 1, 2, 0]}),
            ("oversized", (1, 1, 1, 1), {"kernel_shape": [3, 3]})]:
            for include in ([False, True] if operation == "AveragePool" else [False]):
                if operation == "AveragePool":
                    attrs = dict(attrs, count_include_pad=int(include))
                x = -np.abs(data(shape)) - 1
                add(f"{prefix}_{label}" + ("_include" if include else ""), operation, {"x": x}, **attrs)
        add(prefix + "_nonfinite", operation, {"x": f([np.nan, -np.inf, np.inf, -2]).reshape(1, 1, 2, 2)}, kernel_shape=[2, 2])
    add("max_dilated", "MaxPool", {"x": data((1, 2, 5, 6))}, kernel_shape=[2, 2], dilations=[2, 3], pads=[1, 2, 0, 1])
    add("max_dilated_no_samples", "MaxPool", {"x": data((1, 1, 1, 1))}, kernel_shape=[2, 2], dilations=[2, 2], pads=[1, 1, 1, 1])
    add("max_same_dilated_ceil", "MaxPool", {"x": data((1, 1, 4, 5))}, kernel_shape=[2, 2], dilations=[2, 3], strides=[2, 2], auto_pad="SAME_UPPER", ceil_mode=1)
    add("max_empty_optional_output", "MaxPool", {"x": data((1, 1, 3, 3))}, kernel_shape=[2, 2], empty_output=True)
    for label, x in [("default", data((3, 2, 4, 5))), ("empty_batch", data((0, 2, 3, 3))),
                     ("empty_spatial", data((1, 2, 0, 3))), ("nonfinite", f([np.nan, np.inf]).reshape(1, 1, 1, 2))]:
        add("global_"+label, "GlobalAveragePool", {"x": x})
    for rank in [2, 4]:
        x = data((3, 2) if rank == 2 else (3, 2, 2, 3))
        feeds = {"x": x, "scale": f([2, -1]), "bias": f([0.5, 1]), "mean": f([1, -0.5]), "variance": f([4, 0.25])}
        for label, attrs in [("default", {}), ("epsilon", {"epsilon": 0.125}), ("zero_epsilon", {"epsilon": 0.0}),
                             ("momentum_zero", {"momentum": 0.0}), ("momentum_one", {"momentum": 1.0})]:
            add(f"bn_rank{rank}_{label}", "BatchNormalization", feeds, **attrs)
    for label, x, variance, epsilon in [
        ("zero_domain", f([1, 3, 5]).reshape(3, 1), f([0]), 0.0),
        ("negative_variance", f([1, 3]).reshape(2, 1), f([-1]), 0.0),
        ("empty_batch", data((0, 1)), f([4]), 1e-5),
        ("nonfinite", f([np.nan, np.inf, -np.inf]).reshape(3, 1), f([4]), 1e-5)]:
        add("bn_"+label, "BatchNormalization", {"x": x, "scale": f([2]), "bias": f([0.5]), "mean": f([3]), "variance": variance}, epsilon=epsilon)

    # One symbolic-batch CNN, no intermediate value_info and immutable weights.
    weights = {"w": data((2, 1, 2, 2)), "b": f([0.25, -0.5]), "scale": f([2, -1]),
               "bias": f([0.5, 1]), "mean": f([1, -0.5]), "variance": f([4, 0.25]),
               "dense": data((8, 3)), "offset": f([0.25, -0.5, 0.75])}
    nodes = [helper.make_node("Conv", ["features", "w", "b"], ["conv"], name="convolution", auto_pad="SAME_LOWER"),
             helper.make_node("BatchNormalization", ["conv", "scale", "bias", "mean", "variance"], ["normalized"], name="stored_statistics", epsilon=0.125),
             helper.make_node("Relu", ["normalized"], ["activated"], name="activation"),
             helper.make_node("MaxPool", ["activated"], ["pooled"], name="pool", kernel_shape=[2, 2], strides=[2, 2]),
             helper.make_node("Flatten", ["pooled"], ["flat"], name="flatten"),
             helper.make_node("Gemm", ["flat", "dense", "offset"], ["logits"], name="classifier")]
    graph = helper.make_graph(nodes, "cnn", [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", 1, 4, 4])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 3]),
         helper.make_tensor_value_info("pooled", TensorProto.FLOAT, ["batch", 2, 2, 2])],
        [numpy_helper.from_array(array, name=key) for key, array in weights.items()])
    model = helper.make_model(graph, producer_name="itsmontoya/onnx RT-13", ir_version=8, opset_imports=[helper.make_opsetid("", 13)])
    checker.check_model(model, full_check=True)
    for batch in [1, 3]:
        feeds = {"features": np.arange(1, batch*16+1, dtype=np.float32).reshape(batch, 1, 4, 4) / 8}
        conv = convolution(feeds["features"], weights["w"], weights["b"], auto_pad="SAME_LOWER")
        normalized = normalization({"x": conv, **{key: weights[key] for key in ["scale", "bias", "mean", "variance"]}}, epsilon=0.125)
        pooled = pooling("MaxPool", np.maximum(normalized, 0), kernel_shape=[2, 2], strides=[2, 2])
        expected = {"logits": pooled.reshape(batch, 8) @ weights["dense"] + weights["offset"], "pooled": pooled}
        add(f"cnn_batch_{batch}", "CNN", feeds, expected, model=model)
        if not cases[-1]["observations"]["ort"]["matches_independent"]:
            raise ValueError("composed CNN disagrees with ORT")
    manifest = {"format_version": 1, "onnx_source_revision": REVISION, "reference_versions": VERSIONS,
                "original_license": LICENSE, "cases": cases}
    artifacts["m3/provenance.json"] = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    for path, payload in artifacts.items():
        destination = ROOT / "testdata" / path
        if args.write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        if not destination.exists() or destination.read_bytes() != payload:
            raise SystemExit("M3 fixture differs: " + path)
    existing = {str(path.relative_to(ROOT / "testdata")) for path in (ROOT / "testdata/m3").rglob("*") if path.is_file() and path.suffix != ".md"}
    if existing != set(artifacts):
        raise SystemExit("unexpected/missing M3 artifacts: " + repr(existing ^ set(artifacts)))
    print("PASS: M3", len(cases), dict(Counter(case["status"] for case in cases)),
          "ORT matches", sum(case["observations"]["ort"]["matches_independent"] for case in cases))


if __name__ == "__main__":
    main()
