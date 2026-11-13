"""Controlled post-retention overlays; production sources stay untouched."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass19', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def build(variant):
    sources = {path.name: path.read_text() for path in p.ROOT.glob('*.go') if not path.name.endswith('_test.go')}
    sources['evaluation.go'] = m.command('merged-diagnostic-source-' + variant,
        ['git', 'show', '2445b7920781d6cd91e6635749f6e023044a979b:evaluation.go'])
    if variant == 'arguments':
        sources['evaluation.go'] = replace_once(sources['evaluation.go'], 'symbols map[string]int64',
            'symbols map[string]int64\n arguments [3]*tensor.Tensor')
        old = 'inputs := make([]*tensor.Tensor, len(step.Inputs))'
        assert sources['evaluation.go'].count(old) == 2
        sources['evaluation.go'] = sources['evaluation.go'].replace(old,
            'var inputs []*tensor.Tensor\n if len(step.Inputs) <= len(e.arguments) {\n'
            'inputs = e.arguments[:len(step.Inputs)]\n clear(inputs)\n } else {\n'
            'inputs = make([]*tensor.Tensor, len(step.Inputs))\n }')
    elif variant == 'no-publication':
        sources['evaluation.go'] = replace_once(sources['evaluation.go'],
            'return e.publish(requested)', '_ = requested\n return nil, nil')
    elif variant == 'shell':
        for name in ['numerical_call.go', 'cnn_call.go', 'shape_call.go']:
            source = sources[name]
            start = source.index(') run(ctx context.Context, destination *tensor.Tensor)')
            marker = ('\n\tif c.step.Operator == "Relu" && c.bytes != 0 {'
                if name == 'numerical_call.go' else '\n\tswitch c.step.Operator {')
            position = source.index(marker, start)
            sources[name] = source[:position] + '\n return out, ctx.Err()\n' + source[position:]
    elif variant == 'direct-retention':
        sources['destination_storage.go'] = replace_once(sources['destination_storage.go'],
            'candidate == nil || candidate.DataType() != dataType || !slices.Equal(candidate.Shape(), shape)',
            'candidate == nil')
        sources['destination_storage.go'] = sources['destination_storage.go'].replace('\n\t"slices"\n', '\n')
    elif variant == 'no-output-validation':
        marker = 'func (e *evaluation) checkValue(value operators.Value, backend *tensor.Tensor, kind error) (err error) {'
        sources['evaluation.go'] = replace_once(sources['evaluation.go'], marker, marker + '\n return nil\n')
    elif variant == 'no-onnx-context':
        for name in ['evaluation.go', 'prepared_call.go', 'cnn_call.go', 'numerical_call.go', 'shape_call.go']:
            sources[name] = sources[name].replace('e.ctx.Err()', 'pass19NilError()').replace('ctx.Err()', 'pass19NilError()')
    elif variant != 'control':
        raise ValueError(variant)
    overlay = {str(p.ROOT / ('pass19_' + path.name.removesuffix('.txt'))): str(path)
        for path in HERE.glob('*.go.txt') if path.name != 'backend_test.go.txt'}
    for name, source in sources.items():
        if source == (p.ROOT / name).read_text():
            continue
        target = m.OUT / (variant + '-' + name)
        target.write_text(source)
        m.command('format-' + variant + '-' + name, ['gofmt', '-w', target])
        overlay[str(p.ROOT / name)] = str(target)
    target = m.OUT / (variant + '-overlay.json')
    target.write_text(json.dumps({'Replace': overlay}, indent=2) + '\n')
    m.command('build-' + variant, ['go', 'test', '-c', '-overlay=' + str(target), '-o', m.OUT / (variant + '.test'), '.'])


def components():
    for path in HERE.glob('*.go.txt'):
        m.command('format-' + path.name, ['gofmt', '-w', path])
    build('control')
    m.command('retention-final', [m.OUT / 'control.test', '-test.run=^TestPass19Retention$', '-test.v'])
    m.command('repeat-final', [m.OUT / 'control.test', '-test.run=^TestPass19Repeat$', '-test.v'])
    m.bench('state-final', 'control', '^BenchmarkPass19State$', '100ms', 7)
    m.bench('prepared', 'control', '^BenchmarkPass19Prepared$', '200ms', 7)
    m.bench('parts', 'control', '^BenchmarkPass19Parts$', '200ms', 7)


def ablations():
    variants = ['control', 'arguments', 'no-publication', 'shell', 'direct-retention', 'no-output-validation', 'no-onnx-context']
    for variant in variants:
        build(variant)
        if variant in ['control', 'arguments', 'direct-retention']:
            m.command('repeat-' + variant, [m.OUT / (variant + '.test'), '-test.run=^TestPass19Repeat$', '-test.v'])
    for repeat in range(1, 8):
        for variant in variants if repeat % 2 else list(reversed(variants)):
            m.bench(f'{variant}-ablation-{repeat}', variant, '^BenchmarkPass19Run$', '300ms')


def candidate_components():
    overlay = {str(p.ROOT / ('pass19_' + path.name.removesuffix('.txt'))): str(path)
        for path in HERE.glob('*.go.txt') if path.name != 'backend_test.go.txt'}
    source = (HERE / 'benchmark_test.go.txt').read_text().replace(
        'arguments = make([]*tensor.Tensor, len(step.Inputs))', 'arguments = run.nodeArguments(len(step.Inputs))')
    fixture = m.OUT / 'candidate-parts-fixture.go'
    fixture.write_text(source)
    overlay[str(p.ROOT / 'pass19_benchmark_test.go')] = str(fixture)
    target = m.OUT / 'candidate-parts-overlay.json'
    target.write_text(json.dumps({'Replace': overlay}, indent=2) + '\n')
    m.command('candidate-parts-build', ['go', 'test', '-c', '-overlay=' + str(target), '-o', m.OUT / 'candidate-parts.test', '.'])
    m.command('candidate-retention-repeat-ledger', [m.OUT / 'candidate-parts.test', '-test.run=^TestPass19', '-test.v'])
    m.bench('candidate-state', 'candidate-parts', '^BenchmarkPass19State$', '100ms', 7)
    m.bench('candidate-arguments', 'candidate-parts', '^BenchmarkPass19Parts$/^Arguments$', '200ms', 7)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['components', 'ablations', 'candidate_components'])
    globals()[parser.parse_args().phase]()
