#!/usr/bin/env python3
"""Reproduce RT-14 external MLP weights without changing inherited fixtures."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import tempfile

import google.protobuf
import numpy as np
import onnx
from onnx import TensorProto, checker, numpy_helper
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {"python": "3.9.6", "onnx": "1.16.2", "numpy": "1.26.4", "protobuf": "4.25.3"}
REVISION = "3bf92c03a9f27eba3bda1e5b9e63ea20ec213557"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    actual = {"python": platform.python_version(), "onnx": onnx.__version__,
              "numpy": np.__version__, "protobuf": google.protobuf.__version__}
    if actual != VERSIONS:
        raise SystemExit("reference versions differ: " + repr(actual))
    original_path = ROOT / "testdata/inference/models/mlp.onnx"
    original = onnx.load_model(original_path)
    model = copy.deepcopy(original)
    weights = bytearray(b"RT14 weights\x00")
    ranges = []
    for index, value in enumerate(model.graph.initializer):
        payload = numpy_helper.to_array(value).astype("<f4").tobytes()
        offset = len(weights)
        weights.extend(payload)
        value.ClearField("raw_data")
        value.data_location = TensorProto.EXTERNAL
        entries = [("location", "weights.bin"), ("offset", str(offset))]
        # Exercise the runtime's exact-byte-count rule when length is omitted.
        if index != len(model.graph.initializer)-1:
            entries.append(("length", str(len(payload))))
        for key, content in entries:
            entry = value.external_data.add()
            entry.key, entry.value = key, content
        ranges.append({"name": value.name, "location": "weights.bin", "offset": offset,
                       "bytes": len(payload), "length_omitted": index == len(model.graph.initializer)-1,
                       "sha256": hashlib.sha256(payload).hexdigest()})
    artifacts = {"external/models/mlp.onnx": model.SerializeToString(deterministic=True),
                 "external/models/weights.bin": bytes(weights)}
    inherited = json.loads((ROOT / "testdata/inference/provenance.json").read_text())
    cases = copy.deepcopy([case for case in inherited["cases"]
                           if case["model"]["path"] == "inference/models/mlp.onnx"])
    model_record = {"path": "external/models/mlp.onnx",
                    "sha256": hashlib.sha256(artifacts["external/models/mlp.onnx"]).hexdigest()}
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "mlp.onnx").write_bytes(artifacts["external/models/mlp.onnx"])
        (directory / "weights.bin").write_bytes(bytes(weights))
        checker.check_model(str(directory / "mlp.onnx"), full_check=True)
        restored = onnx.load_model(directory / "mlp.onnx")
        for before, after in zip(original.graph.initializer, restored.graph.initializer):
            np.testing.assert_array_equal(numpy_helper.to_array(before), numpy_helper.to_array(after))
        evaluator = ReferenceEvaluator(restored)
        for case in cases:
            feeds = {entry["name"]: numpy_helper.to_array(onnx.load_tensor(
                ROOT / "testdata" / entry["tensor"]["path"])) for entry in case["inputs"]}
            outputs = evaluator.run(None, feeds)
            for entry, actual_output in zip(case["outputs"], outputs):
                expected = numpy_helper.to_array(onnx.load_tensor(ROOT / "testdata" / entry["tensor"]["path"]))
                np.testing.assert_array_equal(actual_output, expected)
            case["name"] = "external_" + case["name"]
            case["origin"] = "tools/check_external.py; RT-11 MLP with identical externalized weights"
            case["model"] = model_record
    provenance = {"format_version": 1, "onnx_source_revision": REVISION,
                  "reference_versions": VERSIONS, "original_license": inherited["original_license"],
                  "source_model": {"path": "inference/models/mlp.onnx",
                                   "sha256": hashlib.sha256(original_path.read_bytes()).hexdigest()},
                  "weights": {"path": "external/models/weights.bin",
                              "sha256": hashlib.sha256(bytes(weights)).hexdigest()},
                  "ranges": ranges, "cases": cases}
    artifacts["external/provenance.json"] = (json.dumps(provenance, indent=2) + "\n").encode()
    for name, payload in artifacts.items():
        destination = ROOT / "testdata" / name
        if args.write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        elif not destination.exists() or destination.read_bytes() != payload:
            raise SystemExit("artifact differs: " + name)
    print("RT-14 external MLP and shared ranges reproduce; batches 1 and 3 match inherited goldens")


if __name__ == "__main__":
    main()
