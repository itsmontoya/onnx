#!/usr/bin/env python3
"""Verify the RT-01 schema selection and original tiny target model."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import google.protobuf
import numpy
import onnx
from onnx import TensorProto, checker, defs, helper

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {"python": "3.9.6", "onnx": "1.16.2", "numpy": "1.26.4", "protobuf": "4.25.3"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate model and provenance")
    args = parser.parse_args()
    actual = {"python": platform.python_version(), "onnx": onnx.__version__,
              "numpy": numpy.__version__, "protobuf": google.protobuf.__version__}
    if actual != VERSIONS:
        raise SystemExit("reference tool versions differ: " + repr(actual))

    manifest = json.loads((ROOT / "capabilities.json").read_text())
    weights = manifest.get("weights", {})
    if weights.get("status") != "implemented" or weights.get("task") != "RT-14" or weights.get("storage") != ["inline", "external"] or weights.get("data_types") != ["float32", "int64"]:
        raise SystemExit("RT-14 weight capability differs")
    if "external_data_until_RT-14" in manifest["unsupported_features"] or "external_checksum_metadata" not in manifest["unsupported_features"]:
        raise SystemExit("RT-14 external restrictions differ")
    profiles = manifest["profiles"]
    if profiles != [{"ir_version": 8, "domain": "", "min_opset": version, "max_opset": version, "status": "partial_runtime"} for version in [13, 17]]:
        raise SystemExit("tested model profiles differ")
    if len(manifest["operators"]) != 50:
        raise SystemExit("exactly two 25-operator profiles are required")
    operators = [entry for entry in manifest["operators"] if entry["min_opset"] == 13]
    if len(operators) != 25 or len({entry["operator"] for entry in operators}) != 25:
        raise SystemExit("initial profile must contain 25 distinct operators")
    for entry in operators:
        schema = defs.get_schema(entry["operator"], 13, entry["domain"])
        if schema.since_version != entry["schema_version"]:
            raise SystemExit("schema mismatch: " + entry["operator"])
        if set(entry["attributes"]) - set(schema.attributes):
            raise SystemExit("unknown schema attribute: " + entry["operator"])
        if entry["min_opset"] != 13 or entry["max_opset"] != 13:
            raise SystemExit("initial import interval must be exactly 13")
        implemented = entry["operator"] in {"Identity", "Constant", "Add", "Sub", "Mul", "Div", "MatMul", "Gemm", "Relu", "Sigmoid", "Tanh", "Softmax", "Shape", "Flatten", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Gather", "Concat", "Conv", "MaxPool", "AveragePool", "GlobalAveragePool", "BatchNormalization"}
        if entry["status"] != ("implemented" if implemented else "planned"):
            raise SystemExit("unexpected implementation status: " + entry["operator"])
        if implemented and entry["task"] != ("RT-07" if entry["operator"] == "Identity" else "RT-12" if entry["operator"] in {"Shape", "Flatten", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Gather", "Concat"} else "RT-13" if entry["operator"] in {"Conv", "MaxPool", "AveragePool", "GlobalAveragePool", "BatchNormalization"} else "RT-09"):
            raise SystemExit("unexpected implementation task: " + entry["operator"])
        for name, attribute in entry["attributes"].items():
            selected = schema.attributes[name]
            if attribute["type"] != onnx.AttributeProto.AttributeType.Name(int(selected.type)):
                raise SystemExit("attribute type mismatch: " + entry["operator"] + "." + name)
            if "default" in attribute:
                actual_default = helper.get_attribute_value(selected.default_value)
                if isinstance(actual_default, bytes):
                    actual_default = actual_default.decode("utf-8")
                if actual_default != attribute["default"]:
                    raise SystemExit("attribute default mismatch: " + entry["operator"] + "." + name)
    constant = next(entry for entry in operators if entry["operator"] == "Constant")
    if constant["attributes"] != {name: {"type": kind, "required": False} for name, kind in [("value", "TENSOR"), ("value_float", "FLOAT"), ("value_floats", "FLOATS"), ("value_int", "INT"), ("value_ints", "INTS")]}:
        raise SystemExit("RT-09 Constant scalar/list/tensor forms differ")
    # The Go resolver tests use these real version changes in isolated profiles;
    # older resolver-only cases do not expand the two audited production profiles.
    if defs.get_schema("Softmax", 12).attributes["axis"].default_value.i != 1:
        raise SystemExit("Softmax-11 default changed")
    if defs.get_schema("Softmax", 13).attributes["axis"].default_value.i != -1:
        raise SystemExit("Softmax-13 default changed")
    if "axes" not in defs.get_schema("Squeeze", 12).attributes:
        raise SystemExit("Squeeze-11 axes attribute changed")
    if "axes" in defs.get_schema("Squeeze", 13).attributes or defs.get_schema("Squeeze", 13).max_input != 2:
        raise SystemExit("Squeeze-13 axes input changed")

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="copy")], "profile_identity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 2])])
    model = helper.make_model(graph, producer_name="itsmontoya/onnx RT-01",
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    checker.check_model(model, full_check=True)
    payload = model.SerializeToString(deterministic=True)
    provenance = {
        "description": "Original deterministic Identity graph; checker acceptance and hand-calculated identity values",
        "origin": "tools/check_profile.py", "license": "Original repository fixture; no upstream model copied",
        "onnx_source_revision": manifest["onnx_source"]["commit"], "reference_versions": VERSIONS,
        "model": "identity.onnx", "sha256": hashlib.sha256(payload).hexdigest(),
        "ir_version": 8, "opsets": {"": 13}, "operators": {"Identity": 13},
        "inputs": [{"name": "x", "dtype": "float32", "shape": ["batch", 2]}],
        "outputs": [{"name": "y", "dtype": "float32", "shape": ["batch", 2]}],
        "hand_calculated_case": {"batch": 1, "x": [1.0, -2.0], "y": [1.0, -2.0]}}
    model_path = ROOT / "testdata/profile/identity.onnx"
    provenance_path = ROOT / "testdata/profile/provenance.json"
    provenance_text = json.dumps(provenance, indent=2) + "\n"
    if args.write:
        model_path.write_bytes(payload)
        provenance_path.write_text(provenance_text)
    if model_path.read_bytes() != payload or provenance_path.read_text() != provenance_text:
        raise SystemExit("profile fixture changed; inspect and regenerate deliberately")
    checker.check_model(str(model_path), full_check=True)
    print("PASS: 25 schema selections/defaults, 25 implemented operator records, version differences, IR 8/opset 13 checker, deterministic fixture " + provenance["sha256"])


if __name__ == "__main__":
    main()
