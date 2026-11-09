#!/usr/bin/env python3
"""Audit/reproduce byte-original Apache-2.0 ONNX corpus and normalization goldens."""
import argparse
import collections
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
from onnx import AttributeProto, TensorProto, checker, defs, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[1]
REVISION = '3bf92c03a9f27eba3bda1e5b9e63ea20ec213557'
ARCHIVE_HASH = 'd32b94c9c6feda1403d340813b8926d5825082f497e1a65285b50faa31aad3c5'
VERSIONS = {'python': '3.9.6', 'onnx': '1.16.2', 'numpy': '1.26.4', 'protobuf': '4.25.3', 'onnxruntime': '1.19.2'}
NEGATIVES = ['test_layer_normalization_default_axis_expanded', 'test_slice',
 'test_expand_dim_changed', 'test_constant_pad', 'test_resize_downsample_scales_nearest',
 'test_where_example', 'test_cast_FLOAT_to_FLOAT16', 'test_simple_rnn_defaults',
 'test_gru_defaults', 'test_lstm_defaults', 'test_quantizelinear', 'test_if',
 'test_sequence_map_identity_1_sequence', 'test_optional_get_element_optional_tensor']


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def value_record(value):
    value = np.asarray(value)
    return {'dtype': str(value.dtype), 'shape': list(value.shape),
            'bits': value.reshape(-1).view(np.uint32).tolist()}


def agrees(left, right):
    return (len(left) == len(right) and all(a.shape == b.shape and a.dtype == b.dtype and
        np.allclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=True) for a, b in zip(left, right)))


def observe(model, feeds, expected, runtime=False):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if runtime:
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                values = ort.InferenceSession(model.SerializeToString(), options,
                    providers=['CPUExecutionProvider']).run(None, feeds)
            else:
                values = ReferenceEvaluator(model).run(None, feeds)
        return {'status': 'returned', 'comparison': 'agreement' if agrees(values, expected) else 'difference',
                'outputs': [value_record(value) for value in values]}
    except Exception as error:
        return {'status': 'rejected', 'error_type': type(error).__name__, 'error': str(error)}


def normalize(x, scale, bias=None, axis=-1, epsilon=1e-5):
    axes = tuple(range(axis % x.ndim, x.ndim))
    mean = np.mean(x, axis=axes, keepdims=True, dtype=np.float32)
    delta = x - mean
    variance = np.mean(delta * delta, axis=axes, keepdims=True, dtype=np.float32)
    inverse = np.float32(1) / np.sqrt(variance + np.float32(epsilon))
    y = (delta * inverse) * scale
    if bias is not None:
        y = y + bias
    return y, mean, inverse


def metadata(model):
    def info(value):
        kind = value.type.WhichOneof('value')
        if kind != 'tensor_type':
            return {'name': value.name, 'kind': kind}
        tensor = value.type.tensor_type
        return {'name': value.name, 'kind': kind, 'dtype': TensorProto.DataType.Name(tensor.elem_type),
                'shape': [d.dim_value if d.HasField('dim_value') else d.dim_param or None for d in tensor.shape.dim]}
    def attribute(value):
        if value.type in [AttributeProto.INT, AttributeProto.FLOAT, AttributeProto.STRING,
                          AttributeProto.INTS, AttributeProto.FLOATS, AttributeProto.STRINGS]:
            raw = helper.get_attribute_value(value)
            if isinstance(raw, bytes):
                raw = raw.decode()
            return {'type': AttributeProto.AttributeType.Name(value.type), 'value': raw}
        return {'type': AttributeProto.AttributeType.Name(value.type), 'sha256': digest(value.SerializeToString())}
    imports = {v.domain: v.version for v in model.opset_import}
    nodes = []
    for index, node in enumerate(model.graph.node):
        try:
            version = defs.get_schema(node.op_type, imports[node.domain], node.domain).since_version
        except Exception:
            version = None
        nodes.append({'index': index, 'name': node.name, 'domain': node.domain, 'operator': node.op_type,
            'schema_version': version, 'inputs': list(node.input), 'outputs': list(node.output),
            'attributes': {v.name: attribute(v) for v in node.attribute}})
    return {'ir_version': model.ir_version, 'opsets': imports, 'nodes': nodes,
            'inputs': [info(v) for v in model.graph.input], 'outputs': [info(v) for v in model.graph.output],
            'initializers': [{'name': v.name, 'dtype': TensorProto.DataType.Name(v.data_type), 'shape': list(v.dims)}
                             for v in model.graph.initializer],
            'external_data': [{'name': v.name, 'metadata': {x.key: x.value for x in v.external_data}}
                              for v in model.graph.initializer if v.data_location == TensorProto.EXTERNAL],
            'functions': len(model.functions), 'training_info': len(model.training_info)}


def unsupported(description, records):
    findings = []
    def add(feature, node=None, detail=None):
        findings.append({'feature': feature, 'node': node, 'detail': detail})
    if description['ir_version'] != 8 or description['opsets'] not in [{'': 13}, {'': 17}]:
        add('profile', detail={'ir_version': description['ir_version'], 'opsets': description['opsets']})
    for role in ['inputs', 'outputs']:
        for value in description[role]:
            if value['kind'] != 'tensor_type':
                add('value_kind', detail={'role': role, **value})
            elif value['dtype'] not in ['FLOAT', 'INT64']:
                add('dtype', detail={'role': role, **value})
    for node in description['nodes']:
        selected = [r for r in records if r['domain'] == node['domain'] and r['operator'] == node['operator']
          and r['min_opset'] <= description['opsets'].get(node['domain'], 0) <= r['max_opset']]
        if not selected:
            add('operator', node['index'], node['operator'])
            continue
        record = selected[0]
        for name in sorted(node['attributes']):
            if name not in record['attributes']:
                add('attribute', node['index'], name)
        for slot, name in enumerate(node['outputs']):
            if slot >= record['outputs']['max'] or name == '' and slot not in record['outputs']['optional_slots']:
                add('output_slot', node['index'], slot)
    return {'count': len(findings), 'failing_nodes': sorted({v['node'] for v in findings if v['node'] is not None}),
            'features': findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    actual = {'python': platform.python_version(), 'onnx': onnx.__version__, 'numpy': np.__version__,
              'protobuf': google.protobuf.__version__, 'onnxruntime': ort.__version__}
    if actual != VERSIONS or digest(args.source_archive.read_bytes()) != ARCHIVE_HASH:
        raise SystemExit('pinned reference tools/archive differ: ' + repr(actual))
    manifest = json.loads((ROOT / 'capabilities.json').read_text())
    records = manifest['operators']
    artifacts, cases, corpus = {}, [], []
    def artifact(path, payload):
        artifacts[path] = payload
        return {'path': path, 'sha256': digest(payload)}
    def tensor_value(name, role, value_name, value):
        return {'name': value_name, 'tensor': artifact(f'normalization/values/{name}/{role}_{value_name}.pb',
            numpy_helper.from_array(value, name=value_name).SerializeToString(deterministic=True))}
    def case(name, model, model_artifact, feeds, expected, origin, license, inputs=None, outputs=None):
        checker.check_model(model, full_check=True)
        schemas = [{'domain': n.domain, 'operator': n.op_type,
            'version': defs.get_schema(n.op_type, next(v.version for v in model.opset_import if v.domain == n.domain), n.domain).since_version}
            for n in model.graph.node]
        observations = {key: observe(model, feeds, expected, runtime) for key, runtime in [('reference', False), ('onnxruntime', True)]}
        if observations['reference'].get('comparison') != 'agreement' or (observations['onnxruntime'].get('comparison') != 'agreement' and name != 'scalar_broadcast'):
            raise SystemExit('normalization oracle difference (record/review before claiming support): ' + name + repr(observations))
        cases.append({'name': name, 'origin': origin, 'license': license, 'status': 'agreement', 'reason': '',
            'ir_version': model.ir_version, 'opsets': {v.domain: v.version for v in model.opset_import},
            'schemas': schemas, 'model': model_artifact,
            'comparison': {'absolute': 1e-5, 'relative': 1e-5, 'signed_zero': False},
            'inputs': inputs if inputs is not None else [tensor_value(name, 'input', k, v) for k, v in feeds.items()],
            'outputs': outputs if outputs is not None else [tensor_value(name, 'output', v.name, result) for v, result in zip(model.graph.output, expected)],
            'observations': observations})
    with tarfile.open(args.source_archive) as archive:
        prefix = 'onnx-' + REVISION + '/'
        members = {m.name[len(prefix):]: m for m in archive.getmembers() if m.isfile()}
        models = [p for p in sorted(members) if p.startswith('onnx/backend/test/data/node/test_layer_normalization_')
                  and p.endswith('/model.onnx') and '_expanded' not in p]
        selected = models + ['onnx/backend/test/data/node/' + n + '/model.onnx' for n in NEGATIVES]
        # Assert the checked license applies to every copied source/data artifact.
        if archive.extractfile(members['LICENSE']).read() != (ROOT / 'third_party/onnx/LICENSE').read_bytes():
            raise SystemExit('upstream Apache-2.0 LICENSE differs')
        source_path = 'onnx/backend/test/case/node/layernormalization.py'
        source = artifact('corpus/source/layernormalization.py', archive.extractfile(members[source_path]).read())
        envelopes = collections.Counter()
        for path in sorted(members):
            if path.startswith('onnx/backend/test/data/') and path.endswith('/model.onnx'):
                model = onnx.load_model_from_string(archive.extractfile(members[path]).read())
                envelopes[(model.ir_version, tuple((v.domain, v.version) for v in model.opset_import))] += 1
        for path in selected:
            payload = archive.extractfile(members[path]).read()
            model = onnx.load_model_from_string(payload)
            name = path.split('/')[-2]
            model_artifact = artifact('corpus/models/' + name + '.onnx', payload)
            description = metadata(model)
            before = unsupported(description, [r for r in records if r['task'] != 'RT-19'])
            after = unsupported(description, records)
            corpus.append({'name': name, 'origin': 'https://github.com/onnx/onnx/blob/' + REVISION + '/' + path,
                'source_path': path, 'source_sha256': digest(payload), 'license': 'Apache-2.0',
                'license_notice': '../../third_party/onnx/LICENSE', 'model': model_artifact, **description,
                'before_rt19': before, 'after_rt19': after,
                'status': 'enabled' if path in models else 'unsupported'})
            if path not in models:
                continue
            feeds, expected, inputs, outputs = {}, [], [], []
            directory = path[:-len('model.onnx')] + 'test_data_set_0/'
            for role, values in [('input', inputs), ('output', outputs)]:
                for tensor_path in sorted(p for p in members if p.startswith(directory + role + '_') and p.endswith('.pb')):
                    tensor_payload = archive.extractfile(members[tensor_path]).read()
                    tensor = TensorProto.FromString(tensor_payload)
                    array = numpy_helper.to_array(tensor)
                    values.append({'name': tensor.name, 'tensor': artifact('corpus/values/' + name + '/' + tensor_path.split('/')[-1], tensor_payload)})
                    if role == 'input':
                        feeds[tensor.name] = array
                    else:
                        expected.append(array)
            node = model.graph.node[0]
            attrs = {v.name: helper.get_attribute_value(v) for v in node.attribute}
            independent = normalize(feeds[node.input[0]], feeds[node.input[1]], feeds[node.input[2]] if len(node.input) > 2 else None,
                attrs.get('axis', -1), attrs.get('epsilon', 1e-5))
            if not agrees(independent, expected):
                raise SystemExit('independent centered population variance disagrees with upstream: ' + name)
            case(name, model, model_artifact, feeds, expected, corpus[-1]['origin'], 'Apache-2.0', inputs, outputs)
    # Original, test-only helper graphs deliberately have a separate licensing record.
    original_license = 'Original test-only fixtures; no repository LICENSE or separate redistribution grant.'
    def original(name, shape, feeds, attrs=None, slots=None, bias=True, branch=False):
        attrs = attrs or {}
        slots = slots or ['y', 'mean', 'inverse']
        names = ['x', 'scale'] + (['bias'] if bias else [])
        nodes = []
        source_name = 'x'
        if branch:
            nodes.append(helper.make_node('Add', ['x', 'residual'], ['sum'], name='residual_join'))
            source_name = 'sum'
        nodes.append(helper.make_node('LayerNormalization', [source_name] + names[1:], slots, name='normalize', **attrs))
        x = feeds['x'] + feeds['residual'] if branch else feeds['x']
        results = normalize(x, feeds['scale'], feeds.get('bias'), attrs.get('axis', -1), attrs.get('epsilon', 1e-5))
        outputs = [(slot, result) for slot, result in zip(slots, results) if slot]
        if branch:
            nodes.extend([helper.make_node('Mul', ['y', 'y'], ['energy'], name='branch_square'),
                          helper.make_node('Add', ['energy', 'y'], ['score'], name='branch_join')])
            outputs.append(('score', results[0] * results[0] + results[0]))
        graph = helper.make_graph(nodes, name,
            [helper.make_tensor_value_info(k, TensorProto.FLOAT, shape if k in ['x', 'residual'] else list(v.shape)) for k, v in feeds.items()],
            [helper.make_tensor_value_info(k, TensorProto.FLOAT, None) for k, v in outputs])
        # Make output ranks explicit while retaining symbolic batch constraints.
        for output, (_, value) in zip(graph.output, outputs):
            output.type.tensor_type.shape.CopyFrom(helper.make_tensor_value_info('', TensorProto.FLOAT,
                ['batch'] + list(value.shape[1:]) if shape and shape[0] == 'batch' else list(value.shape)).type.tensor_type.shape)
        model = helper.make_model(graph, producer_name='RT-19 test-only normalization', ir_version=8,
            opset_imports=[helper.make_opsetid('', 17)])
        model_artifact = artifact('normalization/models/' + name + '.onnx', model.SerializeToString(deterministic=True))
        case(name, model, model_artifact, feeds, [v for k, v in outputs], 'tools/check_corpus.py test-only helper graph', original_license)
    for batch in [1, 3]:
        x = (np.arange(batch * 6, dtype=np.float32).reshape(batch, 2, 3) - 4) / np.float32(4)
        original('residual_normalization_batch_' + str(batch), ['batch', 2, 3],
            {'x': x, 'residual': x / np.float32(2), 'scale': np.array([1, -2, .5], np.float32), 'bias': np.array([.1, .2, -.3], np.float32)}, branch=True)
    original('optional_inverse_no_bias', [2, 3], {'x': np.arange(6, dtype=np.float32).reshape(2, 3), 'scale': np.full((3,), 2, np.float32)}, slots=['y', '', 'inverse'], bias=False)
    original('scalar_broadcast', [2, 3], {'x': np.arange(6, dtype=np.float32).reshape(2, 3), 'scale': np.array(2, np.float32)}, bias=False)
    original('y_only_zero_epsilon', [2, 3], {'x': np.arange(6, dtype=np.float32).reshape(2, 3), 'scale': np.ones((3,), np.float32)}, attrs={'epsilon': 0.0}, slots=['y'], bias=False)
    original('empty_prefix', [0, 3], {'x': np.empty((0, 3), np.float32), 'scale': np.ones((3,), np.float32)}, bias=False)
    original('axis_zero', [2, 3], {'x': np.arange(6, dtype=np.float32).reshape(2, 3), 'scale': np.ones((2, 3), np.float32)}, attrs={'axis': 0}, bias=False)
    original('constant_zero_epsilon', [2, 3], {'x': np.ones((2, 3), np.float32), 'scale': np.ones((3,), np.float32)}, attrs={'epsilon': 0.0}, bias=False)
    original('nonfinite', [2, 3], {'x': np.array([[np.nan, 1, 2], [np.inf, 0, -np.inf]], np.float32), 'scale': np.ones((3,), np.float32)}, bias=False)
    schema = defs.get_schema('LayerNormalization', 17)
    audit = {'operator': schema.name, 'schema_version': schema.since_version,
        'attributes': {n: {'type': AttributeProto.AttributeType.Name(int(a.type)), 'required': a.required,
            'default': helper.get_attribute_value(a.default_value)} for n, a in sorted(schema.attributes.items())},
        'type_constraints': {v.type_param_str: list(v.allowed_type_strs) for v in schema.type_constraints},
        'documentation_sha256': digest(schema.doc.encode()),
        'decision': 'float32 T/U and stash_type=1 only; centered population variance; fused backend composition; no primitive reduction ONNX adapters'}
    entry = next(r for r in records if r['operator'] == 'LayerNormalization')
    if entry['attributes'] != audit['attributes'] or entry['schema_version'] != 17 or entry['data_types'] != ['float32']:
        raise SystemExit('LayerNormalization manifest/schema differs')
    audit_artifact = artifact('normalization/schema_audit.json', encoded(audit))
    def summary(key):
        counts = collections.Counter(v['feature'] for m in corpus for v in m[key]['features'])
        return {'models': len(corpus), 'enabled': sum(m[key]['count'] == 0 for m in corpus),
                'unsupported': sum(m[key]['count'] > 0 for m in corpus), 'findings': sum(counts.values()), 'features': dict(sorted(counts.items()))}
    artifact('corpus/index.json', encoded({'format_version': 1, 'source_revision': REVISION,
        'archive_sha256': ARCHIVE_HASH, 'license': 'Apache-2.0', 'notice': 'Upstream archive has LICENSE and no NOTICE file; copied unmodified LICENSE retained.',
        'source_export': source, 'upstream_model_count': sum(envelopes.values()),
        'upstream_exact_ir8_opset13': envelopes[(8, (('', 13),))], 'upstream_exact_ir8_opset17': envelopes[(8, (('', 17),))],
        'report_scope': 'Static envelope/value/operator/attribute/output-slot audit of selected corpus; not a universal validator. Go tests assert actual runtime loading and first diagnostics.',
        'before_rt19': summary('before_rt19'), 'after_rt19': summary('after_rt19'), 'models': corpus}))
    artifact('normalization/provenance.json', encoded({'format_version': 1, 'onnx_source_revision': REVISION,
        'reference_versions': VERSIONS, 'original_license': original_license, 'upstream_license': 'Apache-2.0',
        'schema_audit': audit_artifact, 'independent_formula': 'NumPy float32 centered population variance (sum squared deviations / population count); every upstream expected value cross-checked independently.',
        'cases': cases}))
    for path, payload in artifacts.items():
        destination = ROOT / 'testdata' / path
        if args.write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        if not destination.exists() or destination.read_bytes() != payload:
            raise SystemExit('artifact drift: ' + path)
    actual_paths = {str(p.relative_to(ROOT / 'testdata')) for folder in ['corpus', 'normalization']
                    for p in (ROOT / 'testdata' / folder).rglob('*') if p.is_file() and p.name != 'README.md'}
    if actual_paths != set(artifacts):
        raise SystemExit('extra/missing corpus artifacts: ' + repr(actual_paths ^ set(artifacts)))
    print('PASS:', len(corpus), 'byte-original licensed corpus models;', len(cases), 'executed normalization cases (see actual per-oracle observations);', summary('before_rt19'), '->', summary('after_rt19'))


if __name__ == '__main__':
    main()
