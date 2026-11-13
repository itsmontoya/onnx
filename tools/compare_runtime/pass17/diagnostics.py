"""Build test-only ablations; original production files are never edited."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass17', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m


def build(variant):
    evaluation = (p.ROOT / 'evaluation.go').read_text()
    storage = (p.ROOT / 'run_storage.go').read_text()
    if 'arguments' in variant:
        evaluation = evaluation.replace('symbols map[string]int64', 'symbols map[string]int64\n arguments [3]*tensor.Tensor')
        evaluation = evaluation.replace('inputs := make([]*tensor.Tensor, len(step.Inputs))',
            'var inputs []*tensor.Tensor\n if len(step.Inputs) <= len(e.arguments) {\n inputs = e.arguments[:len(step.Inputs)]\n clear(inputs)\n } else {\n inputs = make([]*tensor.Tensor, len(step.Inputs))\n }')
    if 'resolved' in variant:
        evaluation = evaluation.replace('inputs := make([]*tensor.Tensor, len(step.Inputs))',
            'var inputs []*tensor.Tensor\n if pass17UseDestinations {\n inputs = pass17Arguments[step.Index]\n } else {\n inputs = make([]*tensor.Tensor, len(step.Inputs))\n }')
        evaluation = evaluation.replace('if valueIndex >= 0 {\n\t\t\t\tinputs[slot]',
            'if valueIndex >= 0 && !pass17UseDestinations {\n\t\t\t\tinputs[slot]')
    if 'no-publication' in variant:
        evaluation = evaluation.replace('return e.publish(requested)', '_ = requested\n return nil, nil')
    elif 'no-clone' in variant:
        evaluation = evaluation.replace('if value.value, err = source.Clone(); err != nil {',
            'value.value = source\n if err = source.Validate(); err != nil {')
    if 'destinations' in variant:
        storage = storage.replace('candidate := s.free', '''if pass17UseDestinations {
            s.trim()
            if err = s.reserve(bytes, workspace); err != nil { return nil, err }
            out = pass17Destinations[pass17Next]
            pass17Next++
            return out, nil
        }
        candidate := s.free''')
    overlay = {}
    for name, contents in [('evaluation.go', evaluation), ('run_storage.go', storage)]:
        target = m.OUT / (variant + '-' + name)
        target.write_text(contents)
        m.command('format-' + variant + '-' + name, ['gofmt', '-w', target])
        overlay[str(p.ROOT / name)] = str(target)
    if 'no-operators' in variant:
        for name in ['numerical_call.go', 'cnn_call.go', 'shape_call.go']:
            contents = (p.ROOT / name).read_text()
            marker = ('\n\tif c.step.Operator == "Relu" && c.bytes != 0 {'
                      if name == 'numerical_call.go' else '\n\tswitch c.step.Operator {')
            # Shape/CNN files also have planning switches. Modify only run().
            start = contents.index(') run(ctx context.Context, destination *tensor.Tensor)')
            position = contents.index(marker, start)
            contents = contents[:position] + '\n return out, ctx.Err()\n' + contents[position:]
            target = m.OUT / (variant + '-' + name)
            target.write_text(contents)
            m.command('format-' + variant + '-' + name, ['gofmt', '-w', target])
            overlay[str(p.ROOT / name)] = str(target)
    overlay[str(p.ROOT / 'pass17_fixture_test.go')] = str(HERE / 'fixture_test.go.txt')
    target = m.OUT / (variant + '-overlay.json')
    target.write_text(json.dumps({'Replace': overlay}, indent=2) + '\n')
    m.command('build-' + variant, ['go', 'test', '-c', '-overlay=' + str(target), '-o', m.OUT / (variant + '.test'), '.'])


if __name__ == '__main__':
    m.command('format-fixture', ['gofmt', '-w', HERE / 'fixture_test.go.txt'])
    variants = ['control', 'no-clone', 'no-publication', 'arguments', 'destinations',
                'destinations-arguments', 'destinations-arguments-no-publication']
    for variant in variants:
        build(variant)
        m.command('repeat-' + variant, [m.OUT / (variant + '.test'), '-test.run=^TestPass17Repeat$', '-test.v'])
    m.command('ledger', [m.OUT / 'control.test', '-test.run=^TestPass17Ledger$', '-test.v'])
    m.bench('parts', 'control', '^BenchmarkPass17Parts$', '200ms', 7)
    # Interleave each variant with its unchanged control; retain every sample.
    for repeat in range(1, 8):
        for variant in variants if repeat % 2 else list(reversed(variants)):
            m.bench(f'{variant}-diagnostic-{repeat}', variant, '^BenchmarkPass17Run$', '300ms')
