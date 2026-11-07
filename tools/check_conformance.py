#!/usr/bin/env python3
"""Reproduce RT-10 offline fixtures from pinned tools and an official source archive."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import tarfile
import warnings

import google.protobuf
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, defs, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[1]
REVISION = "3bf92c03a9f27eba3bda1e5b9e63ea20ec213557"
ARCHIVE_HASH = "d32b94c9c6feda1403d340813b8926d5825082f497e1a65285b50faa31aad3c5"
VERSIONS = {"python": "3.9.6", "onnx": "1.16.2", "numpy": "1.26.4",
            "protobuf": "4.25.3", "onnxruntime": "1.19.2"}
ORIGINAL_LICENSE = "No repository LICENSE is present; original fixtures have no separately declared license."
UPSTREAM_LICENSE = "Apache-2.0; third_party/onnx/LICENSE; pinned source has no root NOTICE."
COMPARISON = {"absolute": 1e-5, "relative": 1e-5, "signed_zero": True}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encode_json(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def values_record(value):
    value = np.asarray(value)
    payload = (value.reshape(-1).view(np.uint32).tolist() if value.dtype == np.float32
               else value.reshape(-1).tolist())
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "bits" if value.dtype == np.float32 else "integers": payload}


def observe(model, feeds, use_ort=False):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if use_ort:
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                result = ort.InferenceSession(model.SerializeToString(), options,
                                             providers=["CPUExecutionProvider"]).run(None, feeds)
            else:
                result = ReferenceEvaluator(model).run(None, feeds)
        return {"status": "returned", "outputs": [values_record(value) for value in result]}
    except (ValueError, RuntimeError) as error:
        return {"status": "rejected", "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True,
                        help="codeload.github.com/onnx/onnx/tar.gz/" + REVISION)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    actual = {"python": platform.python_version(), "onnx": onnx.__version__, "numpy": np.__version__,
              "protobuf": google.protobuf.__version__, "onnxruntime": ort.__version__}
    if actual != VERSIONS:
        raise SystemExit("reference versions differ: " + repr(actual))
    archive_payload = args.source_archive.read_bytes()
    if digest(archive_payload) != ARCHIVE_HASH:
        raise SystemExit("pinned official source archive hash differs")
    artifacts, cases = {}, []

    def artifact(path, payload):
        if path.startswith("conformance/"):
            artifacts[path] = payload
        elif (ROOT / "testdata" / path).read_bytes() != payload:
            raise ValueError("borrowed artifact differs: " + path)
        return {"path": path, "sha256": digest(payload)}

    def tensor_value(case, role, name, array, raw=False):
        array = np.asarray(array)
        wire = numpy_helper.from_array(array, name=name) if raw else helper.make_tensor(
            name, TensorProto.FLOAT if array.dtype == np.float32 else TensorProto.INT64,
            list(array.shape), array.reshape(-1).tolist())
        path = f"conformance/values/{case}/{role}_{name}.pb"
        return {"name": name, "tensor": artifact(path, wire.SerializeToString(deterministic=True))}

    def add(name, model, model_path, feeds, expected, origin, license_text=ORIGINAL_LICENSE,
            status="agreement", reason="", signed_zero=True, observations=None, raw=False):
        checker.check_model(model, full_check=True)
        imports = {value.domain: value.version for value in model.opset_import}
        schemas = [{"domain": node.domain, "operator": node.op_type,
                    "version": defs.get_schema(node.op_type, imports[node.domain], node.domain).since_version}
                   for node in model.graph.node]
        cases.append({"name": name, "origin": origin, "license": license_text,
                      "status": status, "reason": reason, "ir_version": model.ir_version,
                      "opsets": imports, "schemas": schemas,
                      "model": artifact(model_path, model.SerializeToString(deterministic=True)),
                      "comparison": dict(COMPARISON, signed_zero=signed_zero),
                      "inputs": [tensor_value(name, "input", key, value, raw) for key, value in feeds.items()],
                      "outputs": [tensor_value(name, "output", key, value) for key, value in expected.items()],
                      "observations": observations or {}})

    def original(name, nodes, feeds, expected, status="agreement", reason="", opset=13,
                 output_type=TensorProto.FLOAT, observations=False, raw=False):
        graph = helper.make_graph(nodes, name,
            [helper.make_tensor_value_info(key, TensorProto.FLOAT if value.dtype == np.float32 else TensorProto.INT64,
                                           list(value.shape)) for key, value in feeds.items()],
            [helper.make_tensor_value_info(key, output_type, list(value.shape)) for key, value in expected.items()])
        model = helper.make_model(graph, producer_name="itsmontoya/onnx RT-10",
                                  ir_version=8, opset_imports=[helper.make_opsetid("", opset)])
        result = observe(model, feeds)
        if status == "agreement":
            reference = ReferenceEvaluator(model).run(None, feeds)
            for value, want in zip(reference, expected.values()):
                np.testing.assert_array_equal(value, want)
                if value.shape != want.shape:
                    raise ValueError("hand-calculated shape differs")
        results = {"reference": result, "oracle": "independent literal/formula"}
        if observations:
            results["ort"] = observe(model, feeds, use_ort=True)
        add(name, model, f"conformance/models/{name}.onnx", feeds, expected,
            "tools/check_conformance.py; original graph and independently calculated expectations",
            status=status, reason=reason, observations=results, raw=raw)

    # Reuse RT-09's actual models/expectations; this task adds typed TensorProto values.
    m1 = json.loads((ROOT / "testdata/m1/provenance.json").read_text())
    for case in m1["cases"]:
        def arrays(records):
            result = {}
            for value in records:
                array = (np.asarray(value["bits"], np.uint32).view(np.float32)
                         if value["dtype"] == "float32" else np.asarray(value["integers"], np.int64))
                result[value["name"]] = array.reshape(value["shape"])
            return result
        path = "m1/" + case["model"]
        payload = (ROOT / "testdata" / path).read_bytes()
        if digest(payload) != case["sha256"]:
            raise ValueError("M1 hash differs")
        model = onnx.load_model_from_string(payload)
        feeds, expected = arrays(case["inputs"]), arrays(case["outputs"])
        reference = observe(model, feeds)
        if case["reference_status"] == "agreement":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = ReferenceEvaluator(model).run(None, feeds)
            for value, want in zip(values, expected.values()):
                np.testing.assert_array_equal(value, want)
        elif reference.get("error") != case["reference_error"]:
            raise ValueError("M1 reference exception differs")
        add("m1_" + case["name"], model, path, feeds, expected, m1["origin"],
            status="extension" if case["reference_error"] else "agreement",
            reason=case["reference_error"] or "",
            signed_zero=not case["name"].startswith(("gemm_", "matmul_")) or case["name"] == "matmul_empty_contraction",
            observations={"reference": reference})

    f = lambda value: np.asarray(value, np.float32)
    i = lambda value: np.asarray(value, np.int64)
    node = lambda op, inputs, outputs=("y",), **attrs: helper.make_node(op, inputs, list(outputs), **attrs)
    original("hand_add", [node("Add", ["left", "right"])],
             {"left": f([1, 2]), "right": f([3, 4])}, {"y": f([4, 6])})
    original("hand_branch", [node("Add", ["left", "right"], ("sum",)),
                             node("Mul", ["sum", "right"], ("product",))],
             {"left": f([1, 2]), "right": f([3, 4])}, {"product": f([12, 24]), "sum": f([4, 6])})
    exact = i([16777217, -9223372036854775808, 9223372036854775807])
    original("hand_exact_int64", [node("Identity", ["exact"])], {"exact": exact}, {"y": exact},
             output_type=TensorProto.INT64, raw=True)
    empty = np.empty((2, 0, 3), np.int64)
    original("hand_empty_int64", [node("Identity", ["empty"])], {"empty": empty}, {"y": empty},
             output_type=TensorProto.INT64)
    special = np.array([0x80000000, 0, 0x7f800000, 0xff800000, 0x7fc00001], np.uint32).view(np.float32)
    original("hand_special_identity", [node("Identity", ["special"])], {"special": special}, {"y": special})
    original("hand_scalar", [node("Identity", ["scalar"])], {"scalar": f(-0.0)}, {"y": f(-0.0)})

    # Future adapters remain rejections. Preserve actual reference differences now.
    original("deferred_gather_empty", [node("Gather", ["data", "indices"], axis=1)],
             {"data": np.arange(6, dtype=np.float32).reshape(2, 3), "indices": np.empty((0, 2), np.int64)},
             {"y": np.empty((2, 0, 2), np.float32)}, status="difference",
             reason="RT-12 applies exact prefix/indices/suffix shape [2,0,2]; evaluator incorrectly returns [0]. NumPy take independently confirms expected shape.")
    original("deferred_pool_ceil", [node("AveragePool", ["data"], kernel_shape=[2, 2], strides=[2, 2],
                                        ceil_mode=1, count_include_pad=1)],
             {"data": np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)},
             {"y": f([3, 2.25, 3.75, 2.25]).reshape(1, 1, 2, 2)}, status="unsupported",
             reason="AveragePool-11 adapter is RT-13; full kernel divisors include partial ceil windows.", observations=True)
    original("deferred_pool_nan", [node("MaxPool", ["data"], kernel_shape=[2, 2])],
             {"data": f([np.nan, 1, 2, 3]).reshape(1, 1, 2, 2)}, {"y": f([np.nan]).reshape(1, 1, 1, 1)},
             status="unsupported", reason="MaxPool-12 adapter is RT-13; reference filters input NaNs.", observations=True)
    normalization_feeds = {"data": f([[3], [5]]), "scale": f([2]), "bias": f([0.5]),
                           "mean": f([1]), "variance": f([4])}
    original("deferred_normalization", [node("BatchNormalization", list(normalization_feeds), epsilon=0.0)],
             normalization_feeds, {"y": f([[2.5], [4.5]])}, status="unsupported",
             reason="BatchNormalization-9 adapter is RT-13; stored-statistics formula and ORT agree; evaluator applies momentum.",
             observations=True)
    zero_feeds = {"data": f([[-1], [0], [1]]), "scale": f([1]), "bias": f([0]),
                  "mean": f([0]), "variance": f([0])}
    original("deferred_normalization_zero", [node("BatchNormalization", list(zero_feeds), epsilon=0.0)],
             zero_feeds, {"y": f([[-np.inf], [np.nan], [np.inf]])}, status="unsupported",
             reason="BatchNormalization-9 adapter is RT-13; zero variance/epsilon literal formula differs from ORT's three NaNs.",
             observations=True)
    original("deferred_reduction18", [node("ReduceMean", ["data"], noop_with_empty_axes=1)],
             {"data": f([1, 2])}, {"y": f([1, 2])}, status="unsupported", opset=18,
             reason="NN-12 reduction contract uses opset18; no opset13 reduction adapter/coverage is implied.")

    with tarfile.open(args.source_archive) as archive:
        prefix = "onnx-" + REVISION + "/"
        source_files = {member.name[len(prefix):]: member for member in archive.getmembers() if member.isfile()}
        upstream_models, counts = [], Counter()
        for path, member in sorted(source_files.items()):
            if not path.startswith("onnx/backend/test/data/") or not path.endswith("/model.onnx"):
                continue
            payload = archive.extractfile(member).read()
            model = onnx.load_model_from_string(payload)
            imports = {value.domain: value.version for value in model.opset_import}
            counts[(model.ir_version, json.dumps(imports, sort_keys=True))] += 1
            upstream_models.append({"path": path, "sha256": digest(payload), "ir_version": model.ir_version, "opsets": imports})
        compatible = [model for model in upstream_models if model["ir_version"] == 8 and model["opsets"] == {"": 13}]
        if len(upstream_models) != 1422 or compatible:
            raise ValueError("pinned backend envelope audit differs")
        audit = {"source_revision": REVISION, "archive_sha256": ARCHIVE_HASH, "models": upstream_models,
                 "total": len(upstream_models), "ir8_opset13": 0,
                 "counts": [{"ir_version": ir, "opsets": json.loads(imports), "count": count}
                            for (ir, imports), count in sorted(counts.items())]}
        artifact("conformance/upstream_audit.json", encode_json(audit))
        for name in ["test_softmax_axis_0", "test_gemm_default_no_bias", "test_tanh"]:
            directory = "onnx/backend/test/data/node/" + name + "/"
            payload = archive.extractfile(source_files[directory + "model.onnx"]).read()
            model = onnx.load_model_from_string(payload)
            if model.SerializeToString(deterministic=True) != payload:
                raise ValueError("upstream model would be modified by serialization")
            feeds, expected = {}, {}
            for path, member in sorted(source_files.items()):
                if not path.startswith(directory + "test_data_set_0/") or not path.endswith(".pb"):
                    continue
                tensor_payload = archive.extractfile(member).read()
                wire = TensorProto.FromString(tensor_payload)
                values = feeds if Path(path).name.startswith("input_") else expected
                values[wire.name] = numpy_helper.to_array(wire)
            case_name = "upstream_" + name
            reference_values = ReferenceEvaluator(model).run(None, feeds)
            for value, want in zip(reference_values, expected.values()):
                np.testing.assert_allclose(value, want, atol=1e-5, rtol=1e-5)
                if value.shape != want.shape:
                    raise ValueError("upstream reference shape differs")
            add(case_name, model, f"conformance/upstream/{name}/model.onnx", feeds, expected,
                "https://github.com/onnx/onnx/tree/" + REVISION + "/" + directory,
                UPSTREAM_LICENSE, status="unsupported", reason="Unmodified upstream case has IR7; runtime accepts exactly IR8 (opset13 is compatible).",
                observations={"reference": observe(model, feeds)})
            # Preserve upstream model AND value bytes; generated manifests refer to originals.
            case = cases[-1]
            for role in ["inputs", "outputs"]:
                for value in case[role]:
                    source_path = next(path for path, member in source_files.items()
                        if path.startswith(directory + "test_data_set_0/") and path.endswith(".pb")
                        and TensorProto.FromString(archive.extractfile(member).read()).name == value["name"]
                        and Path(path).name.startswith("input_" if role == "inputs" else "output_"))
                    generated_path = value["tensor"]["path"]
                    del artifacts[generated_path]
                    value["tensor"] = artifact("conformance/upstream/" + name + "/" + Path(source_path).name,
                                               archive.extractfile(source_files[source_path]).read())
        if archive.extractfile(source_files["LICENSE"]).read() != (ROOT / "third_party/onnx/LICENSE").read_bytes():
            raise ValueError("upstream license differs")

    manifest = {"format_version": 1, "onnx_source_revision": REVISION, "reference_versions": VERSIONS,
                "original_license": ORIGINAL_LICENSE, "upstream_license": UPSTREAM_LICENSE,
                "upstream_audit": artifact("conformance/upstream_audit.json", encode_json(audit)),
                "cases": cases}
    artifacts["conformance/provenance.json"] = encode_json(manifest)
    for path, payload in artifacts.items():
        destination = ROOT / "testdata" / path
        if args.write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        if not destination.exists() or destination.read_bytes() != payload:
            raise SystemExit("conformance artifact differs: " + path)
    existing = {str(path.relative_to(ROOT / "testdata")) for path in (ROOT / "testdata/conformance").rglob("*")
                if path.is_file() and path.suffix != ".md"}
    if existing != set(artifacts):
        raise SystemExit("unexpected or missing conformance artifacts: " + repr(existing ^ set(artifacts)))
    print("PASS:", dict(Counter(case["status"] for case in cases)), "; 1422 unchanged upstream models audited; zero IR8/opset13")


if __name__ == "__main__":
    main()
