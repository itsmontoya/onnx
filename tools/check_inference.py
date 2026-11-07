#!/usr/bin/env python3
"""Reproduce original RT-11 dynamic-batch models and actual reference goldens."""
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

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {"python": "3.9.6", "onnx": "1.16.2", "numpy": "1.26.4", "protobuf": "4.25.3"}
REVISION = "3bf92c03a9f27eba3bda1e5b9e63ea20ec213557"
LICENSE = "No repository LICENSE is present; original fixtures have no separately declared license."


def capability_table():
    manifest = json.loads((ROOT / "capabilities.json").read_text())
    rows = ["| Operator (schema) | Types | Inputs → outputs | Attributes | Restrictions |",
            "| --- | --- | --- | --- | --- |"]
    for entry in manifest["operators"]:
        if entry["status"] != "implemented":
            continue
        def arity(value):
            count = str(value["min"]) if value["min"] == value["max"] else f'{value["min"]}–{value["max"]}'
            return count + (f'; optional slots {value["optional_slots"]}' if value["optional_slots"] else "")
        attributes = []
        for name, value in entry["attributes"].items():
            default = f'={value["default"]}' if "default" in value else ""
            attributes.append(f'`{name}:{value["type"]}{default}`')
        rows.append(f'| {entry["operator"]} ({entry["schema_version"]}) | {", ".join(entry["data_types"])} | '
                    f'{arity(entry["inputs"])} → {arity(entry["outputs"])} | {", ".join(attributes) or "None"} | '
                    f'{"; ".join(entry["restrictions"])} |')
    return "\n".join(rows)


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

    def tensor_value(case, role, name, array):
        wire = numpy_helper.from_array(array, name=name)
        return {"name": name, "tensor": artifact(
            f"inference/values/{case}/{role}_{name}.pb", wire.SerializeToString(deterministic=True))}

    f = lambda values: np.asarray(values, dtype=np.float32)
    features = f([[1, 2], [-1, 0.5], [0, -2]])
    residual = f([[0.25, -0.5], [1, 0], [-0.25, 0.75]])
    weights = {"hidden_weight": f([[1, -1, 0.5], [0.5, 1, -1]]),
               "hidden_bias": f([0.25, -0.5, 0.75]),
               "output_weight": f([[1, -0.5], [0.5, 1], [-1, 0.25]]),
               "output_bias": f([0.1, -0.2])}
    for name in ["mlp", "branch"]:
        inputs = [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", 2])]
        nodes = [helper.make_node("Gemm", ["features", "hidden_weight", "hidden_bias"], ["hidden"], name="hidden_affine"),
                 helper.make_node("Relu", ["hidden"], ["activated"], name="hidden_relu"),
                 helper.make_node("Gemm", ["activated", "output_weight", "output_bias"], ["logits"], name="output_affine")]
        if name == "branch":
            inputs.append(helper.make_tensor_value_info("residual", TensorProto.FLOAT, ["batch", 2]))
            nodes.extend([helper.make_node("Add", ["logits", "residual"], ["adjusted"], name="residual_join"),
                          helper.make_node("Softmax", ["adjusted"], ["probabilities"], name="probabilities"),
                          helper.make_node("Sub", ["adjusted", "logits"], ["delta"], name="delta_branch")])
            output_names = ["probabilities", "delta"]
        else:
            nodes.append(helper.make_node("Softmax", ["logits"], ["probabilities"], name="probabilities"))
            output_names = ["probabilities"]
        outputs = [helper.make_tensor_value_info(key, TensorProto.FLOAT, ["batch", 2]) for key in output_names]
        graph = helper.make_graph(nodes, name, inputs, outputs,
                                  [numpy_helper.from_array(value, name=key) for key, value in weights.items()])
        model = helper.make_model(graph, producer_name="itsmontoya/onnx RT-11",
                                  opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        checker.check_model(model, full_check=True)
        model_record = artifact(f"inference/models/{name}.onnx", model.SerializeToString(deterministic=True))
        evaluator = ReferenceEvaluator(model)
        for batch in [1, 3]:
            case = f"{name}_batch_{batch}"
            feeds = {"features": features[:batch].copy()}
            if name == "branch":
                feeds["residual"] = residual[:batch].copy()
            expected = evaluator.run(None, feeds)
            # A second, independent NumPy expression checks composition and branch wiring.
            hidden = np.maximum(feeds["features"] @ weights["hidden_weight"] + weights["hidden_bias"], 0)
            logits = hidden @ weights["output_weight"] + weights["output_bias"]
            adjusted = logits + feeds["residual"] if name == "branch" else logits
            exp = np.exp(adjusted - np.max(adjusted, axis=-1, keepdims=True))
            independent = [exp / np.sum(exp, axis=-1, keepdims=True)]
            if name == "branch":
                independent.append(adjusted - logits)
            for returned, calculated in zip(expected, independent):
                np.testing.assert_allclose(returned, calculated, atol=1e-5, rtol=1e-5)
            cases.append({"name": case, "origin": "tools/check_inference.py; original deterministic graph and values",
                          "license": LICENSE, "status": "agreement", "reason": "", "ir_version": 8,
                          "opsets": {"": 13}, "schemas": [{"domain": node.domain, "operator": node.op_type,
                              "version": defs.get_schema(node.op_type, 13, node.domain).since_version} for node in nodes],
                          "model": model_record,
                          "comparison": {"absolute": 1e-5, "relative": 1e-5, "signed_zero": True},
                          "inputs": [tensor_value(case, "input", key, value) for key, value in feeds.items()],
                          "outputs": [tensor_value(case, "output", key, value) for key, value in zip(output_names, expected)],
                          "observations": {"reference": {"status": "returned", "evaluator": "onnx.reference.ReferenceEvaluator"},
                                           "independent": "NumPy affine/ReLU/stable-softmax formula matched reference"}})
    manifest = {"format_version": 1, "onnx_source_revision": REVISION, "reference_versions": VERSIONS,
                "original_license": LICENSE, "cases": cases}
    artifacts["inference/provenance.json"] = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    for path, payload in artifacts.items():
        destination = ROOT / "testdata" / path
        if args.write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        if not destination.exists() or destination.read_bytes() != payload:
            raise SystemExit("inference fixture differs: " + path)
    guide = ROOT / "docs/getting-started.md"
    text = guide.read_text()
    start, end = "<!-- M1 capabilities start -->", "<!-- M1 capabilities end -->"
    generated = text.split(start)[0] + start + "\n\n" + capability_table() + "\n\n" + end + text.split(end)[1]
    if args.write:
        guide.write_text(generated)
    if guide.read_text() != generated:
        raise SystemExit("getting-started M1 table differs from capabilities.json")
    print("PASS: 2 checker-validated IR-8/opset-13 models, 4 actual reference agreements, batches 1 and 3; canonical M1 table")


if __name__ == "__main__":
    main()
