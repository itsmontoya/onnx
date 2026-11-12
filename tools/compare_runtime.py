#!/usr/bin/env python3
"""Compare this Go runtime with official ONNX Runtime on identical tensor inputs."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

import google.protobuf
import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
REVISION = "4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177"
MODEL_DIRECTORY = "validated/vision/classification/mnist"
SOURCE = f"https://github.com/onnx/models/tree/{REVISION}/{MODEL_DIRECTORY}"
ARCHIVE_URL = (f"https://media.githubusercontent.com/media/onnx/models/{REVISION}/"
               f"{MODEL_DIRECTORY}/model/mnist-12.tar.gz")
ARCHIVE_SHA256 = "a53a59dcaca8804a0f6dfda9a3cf2e082979589391dbde73640b60684f1d24e9"
MODEL_SHA256 = "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def profile(model):
    return {"ir_version": model.ir_version,
            "opsets": {value.domain: value.version for value in model.opset_import},
            "nodes": len(model.graph.node),
            "operators": sorted({node.op_type for node in model.graph.node})}


def prepare_mnist(args):
    archive_path = args.archive or args.cache_dir / "mnist-12.tar.gz"
    if not archive_path.exists():
        if args.archive:
            raise ValueError(f"archive does not exist: {archive_path}")
        print(f"Downloading pinned MNIST model: {ARCHIVE_URL}", file=sys.stderr)
        with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
            payload = response.read()
        if digest(payload) != ARCHIVE_SHA256:
            raise ValueError("downloaded MNIST archive SHA-256 differs")
        archive_path.write_bytes(payload)
    payload = archive_path.read_bytes()
    if digest(payload) != ARCHIVE_SHA256:
        raise ValueError("MNIST archive SHA-256 differs")
    # Read only fixed members; never extract archive-controlled paths.
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        original = archive.extractfile("mnist-12/mnist-12.onnx").read()
        input_payload = archive.extractfile("mnist-12/test_data_set_0/input_0.pb").read()
        output_payload = archive.extractfile("mnist-12/test_data_set_0/output_0.pb").read()
    if digest(original) != MODEL_SHA256:
        raise ValueError("MNIST model SHA-256 differs")
    model = onnx.load_model_from_string(original)
    onnx.checker.check_model(model, full_check=True)
    original_profile = profile(model)
    if original_profile["ir_version"] != 7 or original_profile["opsets"] != {"": 12}:
        raise ValueError("unexpected source MNIST profile")
    converted = onnx.version_converter.convert_version(model, 13)
    converted.ir_version = 8
    onnx.checker.check_model(converted, full_check=True)
    model_path = args.cache_dir / "mnist-ir8-opset13.onnx"
    original_path = args.cache_dir / "mnist-original.onnx"
    original_path.write_bytes(original)
    model_path.write_bytes(converted.SerializeToString(deterministic=True))
    sample = onnx.numpy_helper.to_array(onnx.TensorProto.FromString(input_payload))
    expected = onnx.numpy_helper.to_array(onnx.TensorProto.FromString(output_payload))
    name = model.graph.input[0].name
    cases = {"upstream": {name: sample},
             "zeros": {name: np.zeros(sample.shape, np.float32)},
             "ones": {name: np.ones(sample.shape, np.float32)}}
    random = np.random.default_rng(args.seed)
    for index in range(args.random_cases):
        cases[f"random_{index:03d}"] = {name: random.uniform(0, 1, sample.shape).astype(np.float32)}
    source = {"url": SOURCE, "archive_url": ARCHIVE_URL, "revision": REVISION,
              "archive_sha256": ARCHIVE_SHA256, "model_sha256": MODEL_SHA256,
              "license": "MIT (MNIST model README)", "license_url": SOURCE + "/README.md#license",
              "original_profile": original_profile,
              "conversion": "onnx.version_converter: opset 12 -> 13; IR header 7 -> 8",
              "original_model": str(original_path),
              "sample_input_sha256": digest(input_payload),
              "sample_output_sha256": digest(output_payload)}
    return model_path, converted, cases, source, expected


def load_custom(args):
    if not args.inputs:
        raise ValueError("--model requires at least one --inputs NPZ file")
    model_path = args.model.resolve()
    model = onnx.load(model_path, load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    cases = {}
    for index, path in enumerate(args.inputs):
        with np.load(path, allow_pickle=False) as arrays:
            cases[f"{index:03d}_{path.stem}"] = {name: arrays[name] for name in arrays.files}
    return model_path, model, cases, None, None


def encode_tensor(name, array):
    array = np.asarray(array)
    result = {"name": name, "shape": list(array.shape), "dtype": str(array.dtype)}
    if array.dtype == np.float32:
        result["bits"] = array.reshape(-1).view(np.uint32).tolist()
    elif array.dtype == np.int64:
        result["integers"] = array.reshape(-1).tolist()
    else:
        raise ValueError(f"input {name}: only float32/int64 are supported, got {array.dtype}")
    return result


def decode_tensor(record):
    if record["dtype"] == "float32":
        array = np.asarray(record.get("bits", []), np.uint32).view(np.float32)
    elif record["dtype"] == "int64":
        array = np.asarray(record.get("integers", []), np.int64)
    else:
        raise ValueError(f"unexpected Go output dtype: {record['dtype']}")
    return array.reshape(record["shape"])


def compare(actual, expected, atol, rtol):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"passed": False, "error": "dtype or shape differs",
                "actual_shape": list(actual.shape), "expected_shape": list(expected.shape),
                "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype)}
    if expected.dtype == np.int64:
        mismatches = int(np.count_nonzero(actual != expected))
        return {"passed": mismatches == 0, "mismatched_elements": mismatches}
    actual64, expected64 = actual.astype(np.float64), expected.astype(np.float64)
    finite = np.isfinite(actual64) & np.isfinite(expected64)
    difference = np.abs(actual64[finite] - expected64[finite])
    matches = np.asarray((actual64 == expected64) | (np.isnan(actual64) & np.isnan(expected64)))
    matches[finite] = difference <= atol + rtol * np.abs(expected64[finite])
    mismatches = int(np.count_nonzero(~matches))
    return {"passed": mismatches == 0, "mismatched_elements": mismatches,
            "max_absolute_error": float(difference.max()) if difference.size else 0.0,
            "mean_absolute_error": float(difference.mean()) if difference.size else 0.0,
            "nonfinite_elements": int(np.count_nonzero(~finite))}


def create_ort_session(model_path, optimizations):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                                        if optimizations == "disabled"
                                        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


def run_ort(session, cases, warmup, iterations):
    outputs, latencies = {}, {}
    for name, feeds in sorted(cases.items()):
        for _ in range(warmup):
            session.run(None, feeds)
        elapsed = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            values = session.run(None, feeds)
            elapsed.append(time.perf_counter_ns() - started)
        outputs[name] = dict(zip([value.name for value in session.get_outputs()], values))
        latencies[name] = elapsed
    return outputs, latencies


def run_go(model_path, cases, args):
    environment = dict(os.environ, CGO_ENABLED="0", GOMAXPROCS="1", GOWORK="off")
    request = {name: [encode_tensor(key, value) for key, value in sorted(feeds.items())]
               for name, feeds in sorted(cases.items())}
    with tempfile.TemporaryDirectory(prefix="onnx-compare-") as temporary:
        binary = Path(temporary) / "compare-runtime"
        subprocess.run(["go", "build", "-buildvcs=false", "-o", str(binary), "./tools/compare_runtime"],
                       cwd=ROOT, env=environment, check=True)
        completed = subprocess.run([str(binary), "-model", str(model_path),
                                    "-warmup", str(args.warmup), "-iterations", str(args.iterations)],
                                   input=json.dumps(request, allow_nan=False), text=True,
                                   stdout=subprocess.PIPE, cwd=ROOT, env=environment, check=True)
    result = json.loads(completed.stdout)
    outputs = {name: {value["name"]: decode_tensor(value) for value in values}
               for name, values in result["outputs"].items()}
    return result, outputs


def timing_summary(latencies):
    values = np.asarray([value for case in latencies.values() for value in case], np.float64) / 1000
    return {"samples": int(values.size), "median_us": float(np.median(values)),
            "p95_us": float(np.percentile(values, 95)), "minimum_us": float(values.min())}


def describe_inputs(cases):
    return {name: {key: {"dtype": str(value.dtype), "shape": list(value.shape),
                         "sha256": digest(value.tobytes(order="C"))}
                   for key, value in sorted(feeds.items())} for name, feeds in sorted(cases.items())}


def make_report(args, model_path, model, cases, source, expected):
    started = time.perf_counter_ns()
    session = create_ort_session(model_path, args.ort_optimizations)
    ort_load_ns = time.perf_counter_ns() - started
    original_session = create_ort_session(source["original_model"], "disabled") if source else None
    go_result, go_outputs = run_go(model_path, cases, args)
    ort_outputs, ort_latencies = run_ort(session, cases, args.warmup, args.iterations)
    results = []
    for case_name, reference in sorted(ort_outputs.items()):
        actual = go_outputs[case_name]
        if set(actual) != set(reference):
            raise ValueError(f"case {case_name}: output names differ")
        original = (dict(zip(reference, original_session.run(list(reference), cases[case_name])))
                    if original_session else None)
        for name, array in reference.items():
            result = {"case": case_name, "output": name,
                      "shape": list(array.shape), "dtype": str(array.dtype),
                      "go_vs_ort": compare(actual[name], array, args.atol, args.rtol)}
            if source:
                result["converted_vs_original_ort"] = compare(array, original[name], args.atol, args.rtol)
                if case_name == "upstream":
                    result["ort_vs_upstream_expected"] = compare(original[name], expected, args.atol, args.rtol)
                result["go_top1"] = int(np.argmax(actual[name]))
                result["ort_top1"] = int(np.argmax(array))
                result["top1_agrees"] = result["go_top1"] == result["ort_top1"]
                result["go_output"] = encode_tensor(name, actual[name])
                result["ort_output"] = encode_tensor(name, array)
            results.append(result)
    checks = [check["passed"] for result in results for check in result.values()
              if isinstance(check, dict) and "passed" in check]
    passed = all(checks) and all(result.get("top1_agrees", True) for result in results)
    git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return {"format_version": 1, "passed": passed,
            "model": {"path": str(model_path), "sha256": digest(model_path.read_bytes()), **profile(model)},
            "source": source, "inputs": describe_inputs(cases), "results": results,
            "comparison": {"atol": args.atol, "rtol": args.rtol,
                           "float_policy": "abs(go - ort) <= atol + rtol * abs(ort); NaNs match; infinity signs match; signed zero ignored",
                           "int64_policy": "exact equality"},
            "timing": {"warmup_per_case": args.warmup, "iterations_per_case": args.iterations,
                       "go": timing_summary(go_result["latencies_ns"]),
                       "ort": timing_summary(ort_latencies), "go_load_ns": go_result["load_ns"],
                       "go_session_ns": go_result["session_ns"], "ort_session_load_ns": ort_load_ns,
                       "go_latencies_ns": go_result["latencies_ns"], "ort_latencies_ns": ort_latencies,
                       "scope": "Session.Run / session.run with owned outputs; no build, process startup, model loading, input encoding or output serialization; runtimes measured sequentially"},
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                            "protobuf": google.protobuf.__version__, "platform": platform.platform(),
                            "processor": platform.processor(), "go": subprocess.check_output(["go", "version"], text=True).strip(),
                            "runtime_revision": git_revision, "working_tree_dirty": dirty,
                            "driver_sha256": digest(Path(__file__).read_bytes()),
                            "runner_sha256": {path.name: digest(path.read_bytes())
                                              for path in sorted((ROOT / "tools/compare_runtime").glob("*.go"))},
                            "go_flags": os.environ.get("GOFLAGS", ""),
                            "ort_providers": session.get_providers(), "ort_optimizations": args.ort_optimizations,
                            "ort_threads": 1, "gomaxprocs": 1, "cgo_enabled": 0,
                            "seed": args.seed, "random_cases": args.random_cases}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="compatible local ONNX model; default: pinned pretrained MNIST")
    parser.add_argument("--inputs", type=Path, action="append", help="NPZ of named input arrays; repeat for multiple cases")
    parser.add_argument("--archive", type=Path, help="local pinned MNIST archive, avoiding download")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/compare-runtime")
    parser.add_argument("--report", type=Path, help="JSON destination; default: CACHE_DIR/report.json")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--random-cases", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--ort-optimizations", choices=["disabled", "all"], default="disabled")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.random_cases < 0 or args.seed < 0:
        parser.error("warmup, random-cases and seed must be nonnegative; iterations must be positive")
    if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error("tolerances must be finite and nonnegative")
    if (args.model and args.archive) or (args.inputs and not args.model):
        parser.error("use --model with --inputs, or the default MNIST mode with optional --archive")
    args.cache_dir = args.cache_dir.resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.cache_dir / "report.json"
    model_path, model, cases, source, expected = load_custom(args) if args.model else prepare_mnist(args)
    report = make_report(args, model_path, model, cases, source, expected)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    maximum_error = max((value["go_vs_ort"].get("max_absolute_error", 0) for value in report["results"]), default=0)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(cases)} cases; max absolute error {maximum_error:.8g}")
    if source:
        matches = sum(value["top1_agrees"] for value in report["results"])
        print(f"Top-1 agreement: {matches}/{len(report['results'])}; original/conversion/upstream checks recorded")
    for name in ["go", "ort"]:
        timing = report["timing"][name]
        print(f"{name}: median {timing['median_us']:.2f} us; p95 {timing['p95_us']:.2f} us ({timing['samples']} runs)")
    print(f"Report: {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"comparison failed: {error}", file=sys.stderr)
        sys.exit(1)
