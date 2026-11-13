"""Build explicit investigation overlays without editing production sources."""
import argparse
import importlib.util
import json
import re
import shutil

from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass20', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m


def replace(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def fixture_overlay():
    paths = [HERE.parent / 'pass19' / name for name in
        ['fixture_test.go.txt', 'benchmark_test.go.txt', 'context_test.go.txt']]
    return {str(p.ROOT / ('pass20_inherited_' + path.name.removesuffix('.txt'))): str(path) for path in paths}


def build(variant):
    evaluation = m.command('merged-diagnostic-source-' + variant,
        ['git', 'show', p.BASELINE + ':evaluation.go'])
    sources = {p.ROOT / 'evaluation.go': evaluation}
    options = []
    overlay = fixture_overlay()
    overlay.update({str(p.ROOT / ('pass20_' + path.name.removesuffix('.txt'))): str(path)
        for path in HERE.glob('*.go.txt')})
    if variant == 'local-dtype':
        source = evaluation
        start, end = source.index('func (e *evaluation) checkValue'), source.index('func (e *evaluation) checkInfo')
        body = source[start:end]
        body = replace(body, '\tfor _, declaration := range value.Declarations {',
            '\tdataType := backend.DataType()\n\tfor _, declaration := range value.Declarations {')
        body = body.replace('backend.DataType(), shape, kind', 'dataType, shape, kind')
        sources[p.ROOT / 'evaluation.go'] = source[:start] + body + source[end:]
    elif variant == 'prepared-proof':
        source = (p.ROOT / 'destination_storage.go').read_text()
        source = replace(source, 'values      []*tensor.Tensor', 'values      []*tensor.Tensor\n prepared []*preparedCall')
        marker = '\tcandidate := s.values[slot]\n'
        source = replace(source, marker, marker + '''
	if candidate != nil && slot < len(s.prepared) && s.prepared[slot] != nil {
		expectedType, expectedShape := s.prepared[slot].pass20Geometry()
		if expectedType != dataType || !slices.Equal(expectedShape, shape) {
			return nil
		}
		return candidate
	}
''')
        marker = '\tbytes := value.ByteSize()\n'
        source = replace(source, marker, '''	if slot < len(s.prepared) && s.prepared[slot] != nil {
		expectedType, expectedShape := s.prepared[slot].pass20Geometry()
		if value.DataType() != expectedType || !slices.Equal(value.Shape(), expectedShape) {
			return
		}
	}

''' + marker)
        sources[p.ROOT / 'destination_storage.go'] = source
        sources[p.ROOT / 'session.go'] = replace((p.ROOT / 'session.go').read_text(),
            '\treturn &s\n', '\ts.destinations.prepared = plan.calls\n\treturn &s\n')
        sources[p.ROOT / 'prepared_call.go'] = (p.ROOT / 'prepared_call.go').read_text() + '''
func (p *preparedCall) pass20Geometry() (dataType DataType, shape []int64) {
	switch {
	case p.cnn != nil:
		return Float32, p.cnn.shape
	case p.numerical != nil:
		return Float32, p.numerical.shape
	default:
		return p.shape.dataType, p.shape.shape
	}
}
'''
    elif variant == 'invalid-direct':
        source = (p.ROOT / 'destination_storage.go').read_text()
        source = replace(source,
            'candidate == nil || candidate.DataType() != dataType || !slices.Equal(candidate.Shape(), shape)',
            'candidate == nil')
        sources[p.ROOT / 'destination_storage.go'] = source.replace('\n\t"slices"\n', '\n')
    elif variant == 'shell':
        for name in ['numerical_call.go', 'cnn_call.go', 'shape_call.go']:
            source = (p.ROOT / name).read_text()
            start = source.index(') run(ctx context.Context, destination *tensor.Tensor)')
            marker = ('\n\tif c.step.Operator == "Relu" && c.bytes != 0 {'
                if name == 'numerical_call.go' else '\n\tswitch c.step.Operator {')
            position = source.index(marker, start)
            sources[p.ROOT / name] = source[:position] + '\n return out, ctx.Err()\n' + source[position:]
    elif variant in ['counter', 'counter-candidate']:
        if variant == 'counter-candidate':
            sources[p.ROOT / 'evaluation.go'] = (p.ROOT / 'evaluation.go').read_text()
        reference = m.OUT / 'backend-reference'
        if not reference.exists():
            shutil.copytree(m.BACKEND, reference)
        for original in m.BACKEND.rglob('*'):
            if original.is_file():
                assert original.read_bytes() == (reference / original.relative_to(m.BACKEND)).read_bytes()
        modfile = m.OUT / 'counter.mod'
        modfile.write_bytes((p.ROOT / 'go.mod').read_bytes())
        modfile.with_suffix('.sum').write_bytes((p.ROOT / 'go.sum').read_bytes())
        m.command('counter-reference', ['go', 'mod', 'edit', '-modfile=' + str(modfile),
            '-replace=github.com/itsmontoya/neuralnetwork=' + str(reference)])
        options.append('-modfile=' + str(modfile))
        for name in ['tensor.go', 'destination.go', 'shape_size.go', 'shape.go', 'conv2d.go', 'pool2d.go',
                     'reshape.go', 'matmul_plan.go']:
            path = reference / 'tensor' / name
            source = path.read_text()
            pattern = r'(?m)^func (?:\(t \*Tensor\) )?(\w+)\([^\n]*\) (?:\([^\n]*\)|\*Tensor) \{'
            names = {'Validate', 'DataType', 'Shape', 'Rank', 'Len', 'ByteSize', 'Clone', 'newTensor',
                'validateTensorType', 'validateDestination', 'validateShapeSize', 'ValidateShape',
                'Conv2DShape', 'Pool2DShape', 'ReshapeShape', 'newMatMulPlan', 'shapesEqual'}
            source = re.sub(pattern, lambda match: match.group(0) +
                ('\n pass20Record("' + match.group(1) + '")' if match.group(1) in names else ''), source)
            if name == 'tensor.go':
                source = source.replace('import "fmt"', 'import ("fmt"; "runtime")')
                source += '''
var Pass20Counts = make(map[string]int)
var Pass20Sites = make(map[string]int)

func pass20Record(name string) {
	Pass20Counts[name]++
	var pcs [16]uintptr
	n := runtime.Callers(3, pcs[:])
	frames := runtime.CallersFrames(pcs[:n])
	frame, _ := frames.Next()
	Pass20Sites[name + " <- " + frame.Function]++
}
'''
            sources[path] = source
        path = reference / 'matrix' / 'matrix.go'
        source = path.read_text()
        pattern = r'(?m)^func \(m \*Matrix\) (Rows|Cols|Shape|Validate|Values|ValuesInto|validate|matrixSize)\([^\n]*\) \([^\n]*\) \{'
        source = re.sub(pattern, lambda match: match.group(0) +
            '\n Pass20Counts["' + match.group(1) + '"]++', source)
        sources[path] = source + '\nvar Pass20Counts = make(map[string]int)\n'
        overlay[str(p.ROOT / 'pass20_counter_test.go')] = str(HERE / 'counter_test.go.txt')
    elif variant != 'control':
        raise ValueError(variant)
    if not variant.startswith('counter'):
        overlay.pop(str(p.ROOT / 'pass20_counter_test.go'), None)
    for path, source in sources.items():
        target = m.OUT / (variant + '-' + path.name)
        target.write_text(source)
        m.command('format-' + target.stem, ['gofmt', '-w', target])
        overlay[str(path)] = str(target)
    path = m.OUT / (variant + '-overlay.json')
    path.write_text(json.dumps({'Replace': overlay}, indent=2) + '\n')
    m.command('build-' + variant, ['go', 'test', '-c', *options, '-overlay=' + str(path), '-o', m.OUT / (variant + '.test'), '.'])


def components():
    build('control')
    m.command('retention-and-bits', [m.OUT / 'control.test', '-test.run=^TestPass19', '-test.v'])
    m.command('inventory', [m.OUT / 'control.test', '-test.run=^TestPass20Inventory$', '-test.v'])
    m.bench('prepared', 'control', '^BenchmarkPass19Prepared$', '150ms', 7)
    m.bench('parts', 'control', '^BenchmarkPass19Parts$/^(CheckValues|RetainedLookup|DirectLookup|Acquisition|Bind|Publish)$', '150ms', 7)
    m.bench('metadata', 'control', '^BenchmarkPass20Metadata$', '150ms', 7)


def counters():
    for variant in ['counter', 'counter-candidate']:
        build(variant)
        m.command('accessor-frequency-' + variant, [m.OUT / (variant + '.test'), '-test.run=^TestPass20Counter$', '-test.v'])


def ablations():
    variants = ['control', 'local-dtype', 'prepared-proof', 'invalid-direct', 'shell']
    for variant in variants:
        build(variant)
        if variant not in ['shell', 'invalid-direct']:
            m.command('checks-' + variant, [m.OUT / (variant + '.test'), '-test.run=TestDestinationStorage|TestPass19', '-test.v'])
    for repeat in range(1, 8):
        for variant in variants if repeat % 2 else list(reversed(variants)):
            m.bench(f'{variant}-ablation-{repeat}', variant, '^BenchmarkPass19Run$', '500ms')
    for variant in ['local-dtype', 'prepared-proof']:
        m.bench(variant + '-components', variant,
            '^BenchmarkPass19Parts$/^(CheckValues|RetainedLookup|Acquisition)$', '150ms', 7)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['components', 'counters', 'ablations'])
    globals()[parser.parse_args().phase]()
