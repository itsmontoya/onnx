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
    operators = manifest["operators"]
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
        if entry["status"] != "planned":
            raise SystemExit("RT-01 does not implement operators")

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
        "description": "Original deterministic Identity graph; ONNX checker acceptance only, runtime not implemented",
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
    print("PASS: 25 schema selections, IR 8/opset 13 checker, deterministic fixture " + provenance["sha256"])


if __name__ == "__main__":
    main()
