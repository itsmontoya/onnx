"""Read-only backend investigation using a byte-identical published source copy."""
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass17', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m
reference = m.OUT / 'backend-reference'

if __name__ == '__main__':
    if not reference.exists():
        shutil.copytree(m.BACKEND, reference)
    for original in m.BACKEND.rglob('*'):
        if original.is_file():
            assert original.read_bytes() == (reference / original.relative_to(m.BACKEND)).read_bytes()
    overlay = {}
    for name, path in [('pass16_test.go', HERE.parent / 'pass16/diagnostic_test.go.txt'),
                       ('pass17_test.go', HERE / 'backend_test.go.txt')]:
        overlay[str(reference / 'tensor' / name)] = str(path)
    m.command('format-backend-diagnostic', ['gofmt', '-w', HERE / 'backend_test.go.txt'])
    path = m.OUT / 'backend-overlay.json'
    path.write_text(json.dumps({'Replace': overlay}, indent=2) + '\n')
    m.command('build-backend', ['go', 'test', '-c', '-overlay=' + str(path), '-o', m.OUT / 'backend.test', './tensor'], cwd=reference)
    m.command('backend-ceilings', [m.OUT / 'backend.test', '-test.run=^TestPass(16|17)', '-test.v'])
    m.command('backend-components', [m.OUT / 'backend.test', '-test.run=^$',
        '-test.bench=BenchmarkPass16MatMul|BenchmarkPass17', '-test.benchtime=200ms', '-test.count=7', '-test.benchmem'])
    for name, options, extra in [
        ('no-cgo', [], {}), ('purego', ['-tags=purego'], {}),
        ('race', ['-race'], {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}),
        ('purego-race', ['-race', '-tags=purego'], {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}),
    ]:
        m.command('backend-diagnostic-' + name, ['go', 'test', '-count=1', '-overlay=' + str(path),
            *options, '-run=^TestPass(16|17)', './tensor'], extra, reference)
