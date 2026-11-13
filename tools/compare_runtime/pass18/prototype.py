"""Build a Session-owned, quota-charged prototype through Go source overlays."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass18', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m
command = m.command
m.command = lambda name, *args, **kwargs: command("v2-" + name, *args, **kwargs)


def build():
    sources = {path.name: path.read_text() for path in (m.OUT / 'original').glob('*.go')}
    sources['destination_storage.go'] = (HERE / 'destination_storage.go.txt').read_text()
    sources['execution_plan.go'] = sources['execution_plan.go'].replace('return &p, nil',
        'p.destinationLimit = p.destinationBudget(ctx)\n return &p, nil', 1).replace('callBytes     int64',
        'callBytes     int64\n destinationLimit int64\n destinationMinimum int64')
    sources['execution_plan.go'] += (HERE / 'destination_budget.go.txt').read_text()
    sources['session.go'] = sources['session.go'].replace('s.plan, s.initialized = plan, true',
        's.plan, s.initialized = plan, true\n s.destinations = newDestinationStorage(plan.destinationLimit, plan.destinationMinimum, len(plan.steps))')
    sources['session.go'] = sources['session.go'].replace('mu          sync.Mutex',
        'mu          sync.Mutex\n destinations destinationStorage')
    sources['session.go'] = sources['session.go'].replace('if outputs, err = run.run(inputs, options);',
        'if s.plan.limits.MaxLiveBytes < s.destinations.minimumLive { s.destinations = destinationStorage{} }\n run.storage.destinations = &s.destinations\n run.budget.live += s.destinations.bytes\n if outputs, err = run.run(inputs, options);')
    sources['session.go'] = sources['session.go'].replace('s.plan.release()',
        's.destinations = destinationStorage{}\n s.plan.release()')
    sources['evaluation.go'] = sources['evaluation.go'].replace('var backend *tensor.Tensor\n\n\tswitch step.Operator',
        'var backend *tensor.Tensor\n e.storage.slot, e.storage.fresh = -1, false\n\n\tswitch step.Operator')
    sources['evaluation.go'] = sources['evaluation.go'].replace('if backend, err = e.plan.calls[step.Index].run(e, inputs);',
        'e.storage.slot = step.Index\n if backend, err = e.plan.calls[step.Index].run(e, inputs);')
    sources['evaluation.go'] = sources['evaluation.go'].replace('e.storage.values[index] = backend\n\treturn nil',
        'if e.storage.fresh { e.storage.destinations.keep(e.storage.slot, backend, e.budget) }\n e.storage.values[index] = backend\n\treturn nil', 1)
    sources['run_storage.go'] = sources['run_storage.go'].replace('reuses    int',
        'reuses    int\n destinations *destinationStorage\n slot int\n fresh bool')
    sources['run_storage.go'] = sources['run_storage.go'].replace('return nil, nil\n}',
        'if out = s.destinations.get(s.slot, dataType, shape); out != nil { return out, nil }\n s.fresh = true\n return nil, nil\n}', 1)
    # Constant prepared Reshape still performs its original reservations, but
    # must neither borrow nor capture mutable Session destination storage.
    sources['prepared_call.go'] = sources['prepared_call.go'].replace('call := *p.shape\n\t\tcall.inputs = inputs',
        'call := *p.shape\n\t\tcall.inputs = inputs\n if p.retained != nil { e.storage.slot = -1 }')
    overlay = {}
    directory = m.OUT / 'prototype'
    directory.mkdir(exist_ok=True)
    for name, contents in sources.items():
        if name not in ['session.go', 'run_storage.go', 'evaluation.go', 'execution_plan.go', 'prepared_call.go', 'destination_storage.go']:
            continue
        path = directory / name
        path.write_text(contents)
        m.command('prototype-format-' + name, ['gofmt', '-w', path])
        overlay[str(p.ROOT / name)] = str(path)
    overlay[str(p.ROOT / 'destination_storage_test.go')] = str(HERE / 'destination_storage_test.go.txt')
    overlay[str(p.ROOT / 'pass18_benchmark_test.go')] = str(HERE / 'benchmark_test.go.txt')
    path = m.OUT / 'prototype-overlay.json'
    path.write_text(json.dumps({'Replace': overlay}, indent=2) + '\n')
    m.command('prototype-build', ['go', 'test', '-c', '-overlay=' + str(path), '-o', m.OUT / 'prototype.test', '.'])
    m.command('prototype-tests', [m.OUT / 'prototype.test', '-test.run=TestDestinationStorage', '-test.v'])
    m.bench('prototype-lifecycles', 'prototype', '^BenchmarkPass18', '200ms', 5)
    for repeat in range(1, 6):
        for variant in ['control', 'prototype'] if repeat % 2 else ['prototype', 'control']:
            m.bench(f'prototype-{variant}-ramp-{repeat}', variant, '^BenchmarkMNIST$', '300ms')


if __name__ == '__main__':
    build()
