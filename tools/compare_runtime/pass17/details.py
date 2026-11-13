"""Complete the current prepared traversal and pre-resolved-argument probes."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('diagnostics', HERE / 'diagnostics.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
m = d.m

if __name__ == '__main__':
    m.command('format-fixture-final', ['gofmt', '-w', HERE / 'fixture_test.go.txt'])
    for variant in ['current-control', 'destinations-resolved', 'destinations-resolved-no-publication']:
        d.build(variant)
        m.command('repeat-' + variant, [m.OUT / (variant + '.test'), '-test.run=^TestPass17Repeat$', '-test.v'])
    m.command('ledger-final', [m.OUT / 'current-control.test', '-test.run=^TestPass17Ledger$', '-test.v'])
    m.bench('prepared-current', 'current-control', '^BenchmarkPass17Prepared$', '200ms', 7)
    for repeat in range(1, 8):
        variants = ['current-control', 'destinations-resolved', 'destinations-resolved-no-publication']
        for variant in variants if repeat % 2 else list(reversed(variants)):
            m.bench(f'{variant}-diagnostic-{repeat}', variant, '^BenchmarkPass17Run$', '300ms')
