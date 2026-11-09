#!/usr/bin/env python3
"""Reproduce RT-17's real PyTorch export, two-profile schema audit and goldens."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import warnings

import google.protobuf
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, checker, defs, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[1]
REVISION = "3bf92c03a9f27eba3bda1e5b9e63ea20ec213557"
VERSIONS = {"python": "3.9.6", "onnx": "1.16.2", "numpy": "1.26.4",
            "protobuf": "4.25.3", "onnxruntime": "1.19.2", "torch": "2.2.2"}
LICENSE = "No repository LICENSE is present; original fixtures have no separately declared license."
EXPORTER_REVISION = "39901f229520a5256505ec24782f716ee7ddc843"
CHANGES = {"Identity": 16, "Add": 14, "Sub": 14, "Mul": 14, "Div": 14,
           "Relu": 14, "Reshape": 14, "Shape": 15, "BatchNormalization": 15}


class ExportedMLP(torch.nn.Module):
    """Original deterministic module; no external model or trained data copied."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.output = torch.nn.Linear(3, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1., .5], [-1., 1.], [.5, -1.]]))
            self.linear.bias.copy_(torch.tensor([.25, -.5, .75]))
            self.output.weight.copy_(torch.tensor([[1., .5, -1.], [-.5, 1., .25]]))
            self.output.bias.copy_(torch.tensor([.1, -.2]))

    def forward(self, features):
        return torch.softmax(self.output(torch.relu(self.linear(features))), dim=-1)


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def values_record(value):
    value = np.asarray(value)
    payload = (value.reshape(-1).view(np.uint32).tolist() if value.dtype == np.float32
               else value.reshape(-1).tolist())
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "bits" if value.dtype == np.float32 else "integers": payload}


def observe(model, feeds, runtime=False):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if runtime:
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                results = ort.InferenceSession(model.SerializeToString(), options,
                    providers=["CPUExecutionProvider"]).run(None, feeds)
            else:
                results = ReferenceEvaluator(model).run(None, feeds)
        return {"status": "returned", "outputs": [values_record(value) for value in results]}, results
    except (ValueError, RuntimeError) as error:
        return {"status": "rejected", "error": str(error)}, None


def compare_outputs(actual, expected):
    if len(actual) != len(expected):
        raise SystemExit("reference output count differs")
    for left, right in zip(actual, expected):
        if left.dtype != right.dtype or left.shape != right.shape:
            raise SystemExit("reference dtype/shape differs")
        if right.dtype == np.int64:
            np.testing.assert_array_equal(left, right)
            continue
        np.testing.assert_allclose(left, right, atol=1e-5, rtol=1e-5)
        zeros = (left == 0) & (right == 0)
        np.testing.assert_array_equal(np.signbit(left[zeros]), np.signbit(right[zeros]))


def schema_record(schema):
    attributes = {}
    for name, attribute in sorted(schema.attributes.items()):
        record = {"type": onnx.AttributeProto.AttributeType.Name(int(attribute.type)),
                  "required": attribute.required}
        if attribute.default_value.name:
            value = helper.get_attribute_value(attribute.default_value)
            record["default"] = value.decode() if isinstance(value, bytes) else value
        attributes[name] = record
    return {"version": schema.since_version, "min_input": schema.min_input,
            "max_input": schema.max_input, "min_output": schema.min_output,
            "max_output": schema.max_output, "attributes": attributes,
            "type_constraints": {value.type_param_str: list(value.allowed_type_strs)
                                 for value in schema.type_constraints},
            "has_function": schema.has_function,
            "has_context_dependent_function": schema.has_context_dependent_function,
            "documentation_sha256": hashlib.sha256(schema.doc.encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    actual = {"python": platform.python_version(), "onnx": onnx.__version__,
              "numpy": np.__version__, "protobuf": google.protobuf.__version__,
              "onnxruntime": ort.__version__, "torch": torch.__version__}
    if actual != VERSIONS or torch.version.git_version != EXPORTER_REVISION:
        raise SystemExit("reference versions differ: " + repr(actual))
    torch.set_num_threads(1)
    torch.manual_seed(0)
    manifest = json.loads((ROOT / "capabilities.json").read_text())
    if manifest["onnx_source"]["commit"] != REVISION or len(manifest["operators"]) != 50:
        raise SystemExit("manifest source or record count differs")
    records = manifest["operators"]
    audit = []
    for old in records[:25]:
        name = old["operator"]
        pair = []
        for imported in [13, 17]:
            entries = [entry for entry in records if entry["operator"] == name and
                       entry["domain"] == "" and entry["min_opset"] == imported and
                       entry["max_opset"] == imported]
            if len(entries) != 1:
                raise SystemExit("profile must have one exact interval for " + name)
            entry = entries[0]
            schema = defs.get_schema(name, imported)
            if schema.since_version != entry["schema_version"]:
                raise SystemExit("applicable schema differs: " + name)
            selected = schema_record(schema)
            for key, attribute in entry["attributes"].items():
                if selected["attributes"].get(key) != attribute:
                    raise SystemExit("attribute/default/required mismatch: " + name + "." + key)
            if entry["data_types"] != old["data_types"]:
                raise SystemExit("profile expanded the bounded value types")
            pair.append(selected)
        if pair[1]["version"] != CHANGES.get(name, pair[0]["version"]):
            raise SystemExit("unexpected schema change: " + name)
        decision = ("allowzero 0 copies dimensions; 1 keeps literal zeros; zero with -1 rejected"
                    if name == "Reshape" else
                    "start/end slice concrete rank with negative indexing and clamping"
                    if name == "Shape" else
                    "training_mode=0 and one output only; all five inputs float32; stored-statistics inference"
                    if name == "BatchNormalization" else
                    "new type/function forms do not expand dense float32/int64 runtime subset; float32 formulas unchanged"
                    if name in CHANGES else "same applicable schema; original restrictions preserved")
        audit.append({"operator": name, "import_13": pair[0], "import_17": pair[1],
                      "runtime_decision": decision})

    artifacts, cases = {}, []

    def artifact(path, payload):
        artifacts[path] = payload
        return {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}

    def tensor_value(case, role, name, value):
        tensor = numpy_helper.from_array(np.asarray(value), name=name)
        return {"name": name, "tensor": artifact(f"profiles/values/{case}/{role}_{name}.pb",
                tensor.SerializeToString(deterministic=True))}

    def add_case(name, model, feeds, expected, status="agreement", reason="", origin=None, independent=None):
        checker.check_model(model, full_check=True)
        imports = {value.domain: value.version for value in model.opset_import}
        reference, returned = observe(model, feeds)
        runtime, ort_returned = observe(model, feeds, True)
        if status == "agreement":
            if returned is None:
                raise SystemExit("unexpected reference failure: " + name + repr(reference))
            compare_outputs(returned, expected)
        elif status == "difference":
            if returned is None or all(np.array_equal(left, right, equal_nan=True)
                                       for left, right in zip(returned, expected)):
                raise SystemExit("expected executed difference did not reproduce: " + name)
        if ort_returned is not None and status != "unsupported":
            compare_outputs(ort_returned, expected)
        cases.append({"name": name, "origin": origin or "tools/check_profiles.py; original deterministic graph and values",
            "license": LICENSE, "status": status, "reason": reason, "ir_version": model.ir_version,
            "opsets": imports, "schemas": [{"domain": node.domain, "operator": node.op_type,
                "version": defs.get_schema(node.op_type, imports[node.domain], node.domain).since_version}
                for node in model.graph.node],
            "model": artifact(f"profiles/models/{name}.onnx", model.SerializeToString(deterministic=True)),
            "comparison": {"absolute": 1e-5, "relative": 1e-5, "signed_zero": True},
            "inputs": [tensor_value(name, "input", key, value) for key, value in feeds.items()],
            "outputs": [tensor_value(name, "output", wire.name, value)
                        for wire, value in zip(model.graph.output, expected)],
            "observations": {"reference": reference, "onnxruntime": runtime,
                "independent": independent or "independent NumPy/literal expectation"}})

    def original(name, imported, operator, feeds, expected, attrs=None, constants=None,
                 node_inputs=None, status="agreement", reason="", output_names=None):
        names = output_names or ["y"]
        inputs = [helper.make_tensor_value_info(key, numpy_helper.from_array(value).data_type,
                                               list(value.shape)) for key, value in feeds.items()]
        outputs = [helper.make_tensor_value_info(key, numpy_helper.from_array(value).data_type,
                                                list(value.shape)) for key, value in zip(names, expected)]
        constants = constants or {}
        node = helper.make_node(operator, node_inputs or list(feeds) + list(constants), names,
                                name="audited_" + operator, **(attrs or {}))
        graph = helper.make_graph([node], name, inputs, outputs,
                [numpy_helper.from_array(value, name=key) for key, value in constants.items()])
        model = helper.make_model(graph, producer_name="itsmontoya/onnx RT-17", ir_version=8,
                                   opset_imports=[helper.make_opsetid("", imported)])
        add_case(name, model, feeds, expected, status, reason)

    f = lambda value: np.asarray(value, np.float32)
    i = lambda value: np.asarray(value, np.int64)
    x, b = f([[1, -2], [3, 4]]), f([2, 4])
    for imported in [13, 17]:
        prefix = f"opset{imported}_"
        original(prefix + "identity", imported, "Identity", {"x": x}, [x])
        exact = i([-(2**63), 2**63-1, 16777217])
        original(prefix + "constant", imported, "Constant", {}, [exact],
                 {"value": numpy_helper.from_array(exact)})
        for operator, result in [("Add", x+b), ("Sub", x-b), ("Mul", x*b), ("Div", x/b)]:
            original(prefix + operator.lower(), imported, operator, {"x": x, "b": b}, [result])
        original(prefix + "matmul", imported, "MatMul", {"x": x, "b": f([[1, 2], [3, 4]])},
                 [x @ f([[1, 2], [3, 4]])])
        original(prefix + "gemm", imported, "Gemm", {"x": x, "b": f([[1, 2], [3, 4]])},
                 [x @ f([[1, 2], [3, 4]])], node_inputs=["x", "b", ""])
        original(prefix + "relu", imported, "Relu", {"x": x}, [np.maximum(x, 0)])
        original(prefix + "sigmoid", imported, "Sigmoid", {"x": x}, [1/(1+np.exp(-x))])
        original(prefix + "tanh", imported, "Tanh", {"x": x}, [np.tanh(x)])
        exp = np.exp(x-np.max(x, axis=-1, keepdims=True))
        original(prefix + "softmax", imported, "Softmax", {"x": x}, [exp/exp.sum(axis=-1, keepdims=True)])
        original(prefix + "flatten", imported, "Flatten", {"x": x.reshape(1, 2, 2)}, [x.reshape(1, 4)])
        original(prefix + "reshape", imported, "Reshape", {"x": x}, [x.reshape(2, 2)], constants={"target": i([0, -1])})
        original(prefix + "transpose", imported, "Transpose", {"x": x}, [x.T])
        original(prefix + "squeeze", imported, "Squeeze", {"x": x.reshape(1, 2, 2)}, [x],
                 constants={"axes": i([0])})
        original(prefix + "unsqueeze", imported, "Unsqueeze", {"x": x}, [x.reshape(1, 2, 2)],
                 constants={"axes": i([0])})
        original(prefix + "shape", imported, "Shape", {"x": x}, [i([2, 2])])
        original(prefix + "gather", imported, "Gather", {"x": exact}, [exact[[2, 0]]], constants={"indices": i([2, 0])})
        original(prefix + "concat", imported, "Concat", {"x": x, "b": x}, [np.concatenate([x, x], axis=0)], {"axis": 0})
        image = f([[[[1, 2], [3, 4]]]])
        original(prefix + "conv", imported, "Conv", {"x": image}, [image*2], constants={"w": f([[[[2]]]])}, node_inputs=["x", "w", ""])
        original(prefix + "maxpool", imported, "MaxPool", {"x": image}, [f([[[[4]]]])], {"kernel_shape": [2, 2]})
        original(prefix + "averagepool", imported, "AveragePool", {"x": image}, [f([[[[2.5]]]])], {"kernel_shape": [2, 2]})
        original(prefix + "globalaveragepool", imported, "GlobalAveragePool", {"x": image}, [f([[[[2.5]]]])])
        # Equal data/stored statistics make both BN evaluators agree in either schema.
        original(prefix + "batchnormalization", imported, "BatchNormalization", {"x": f([[1], [-1]])},
                 [f([[1], [-1]])], {"epsilon": 0.0},
                 constants={"scale": f([1]), "bias": f([0]), "mean": f([0]), "variance": f([1])})
    for imported in [13, 17]:
        for name, value in [("int64", i([-(2**63), 2**63-1, 16777217])),
                            ("empty", np.empty((0, 2), np.int64)),
                            ("special", f([np.nan, np.inf, -np.inf, -0.0, 0.0]))]:
            original(f"identity_{name}_{imported}", imported, "Identity", {"x": value}, [value])
    empty = np.empty((0, 2), np.float32)
    for allowzero in [0, 1]:
        original(f"reshape_allowzero_{allowzero}", 17, "Reshape", {"x": empty},
                 [empty.reshape(0, 2) if allowzero == 0 else empty.reshape(2, 0)],
                 {"allowzero": allowzero}, constants={"target": i([0, 2] if allowzero == 0 else [2, 0])})
    for start, end in [(1, None), (-2, -1), (2, 1), (-(2**63), 2**63-1), (2**63-1, -(2**63))]:
        attrs = {"start": start}
        if end is not None:
            attrs["end"] = end
        original(f"shape_slice_{len(cases)}", 17, "Shape", {"x": f(np.arange(24).reshape(2, 3, 4))},
                 [i([2, 3, 4][start:end])], attrs)
    # BN-15 fixes the evaluator's old momentum application; preserve both observations.
    for imported in [13, 17]:
        original(f"bn_stored_statistics_{imported}", imported, "BatchNormalization", {"x": f([[3], [5]])},
                 [f([[2.5], [4.5]])], {"epsilon": 0.0, **({"training_mode": 0} if imported == 17 else {})},
                 constants={"scale": f([2]), "bias": f([.5]), "mean": f([1]), "variance": f([4])},
                 status="difference" if imported == 13 else "agreement",
                 reason="BN-9 evaluator applies momentum; runtime and ORT use unchanged stored statistics." if imported == 13 else "")

    exported = ExportedMLP().eval()
    data = io.BytesIO()
    torch.onnx.export(exported, torch.tensor([[1., 2.]]), data, opset_version=17,
                      input_names=["features"], output_names=["probabilities"],
                      dynamic_axes={"features": {0: "batch"}, "probabilities": {0: "batch"}},
                      do_constant_folding=False)
    payload = data.getvalue()
    model = onnx.load_from_string(payload)
    if model.ir_version != 8 or [(value.domain, value.version) for value in model.opset_import] != [("", 17)]:
        raise SystemExit("actual exporter does not emit selected IR-8/opset-17 profile")
    # Never rewrite or reserialize the export before storing and hashing it.
    export_record = artifact("profiles/exports/pytorch_mlp.onnx", payload)
    for batch in [1, 3]:
        features = f([[1, 2], [-1, .5], [0, -2]])[:batch].copy()
        with torch.no_grad():
            expected = [exported(torch.from_numpy(features)).numpy()]
        add_case(f"pytorch_mlp_batch_{batch}", model, {"features": features}, expected,
                 origin="Original ExportedMLP above; actual torch.onnx.export 2.2.2; no envelope conversion",
                 independent="PyTorch eager inference compared with ONNX evaluator and ORT")
        artifacts.pop(cases[-1]["model"]["path"])
        cases[-1]["model"] = export_record
    audit_record = artifact("profiles/schema_audit.json", encoded({"onnx_source_revision": REVISION, "operators": audit}))
    provenance = {"format_version": 1, "onnx_source_revision": REVISION, "reference_versions": VERSIONS,
        "original_license": LICENSE, "schema_audit": audit_record, "exporter": {"name": "torch.onnx.export", "version": "2.2.2",
            "source": "https://github.com/pytorch/pytorch/tree/" + EXPORTER_REVISION + "/torch/onnx",
            "source_revision": EXPORTER_REVISION,
            "tool_license": "BSD-3-Clause; optional development dependency, no PyTorch implementation copied",
            "demand": "Execute actual default float32 dynamic-batch MLP inference exports requesting opset 17",
            "unmodified_export": export_record, "ir_version": 8, "opsets": {"": 17},
            "runtime_conversion": False}, "cases": cases}
    artifacts["profiles/schema_audit.json"] = encoded({"onnx_source_revision": REVISION, "operators": audit})
    artifacts["profiles/provenance.json"] = encoded(provenance)
    for path, value in artifacts.items():
        destination = ROOT / "testdata" / path
        if args.write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(value)
        if not destination.exists() or destination.read_bytes() != value:
            raise SystemExit("profile fixture differs: " + path)
    present = {str(path.relative_to(ROOT / "testdata")) for path in (ROOT / "testdata/profiles").rglob("*")
               if path.is_file() and path.name != "README.md"}
    if present != set(artifacts):
        raise SystemExit("extra/missing profile artifacts: " + repr(present ^ set(artifacts)))
    print("PASS: 25 exact schema pairs, unmodified PyTorch IR-8/opset-17 export,", len(cases), "executed cases")


if __name__ == "__main__":
    main()
