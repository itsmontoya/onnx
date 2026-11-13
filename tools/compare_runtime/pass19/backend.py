"""Inspect published backend code through test-only overlays on an exact copy."""
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass19', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m

if __name__ == '__main__':
    reference = m.OUT / 'backend-reference'
    if not reference.exists():
        shutil.copytree(m.BACKEND, reference)
    for original in m.BACKEND.rglob('*'):
        if original.is_file():
            assert original.read_bytes() == (reference / original.relative_to(m.BACKEND)).read_bytes()
    paths = [HERE.parent / 'pass16/diagnostic_test.go.txt',
        HERE.parent / 'pass17/backend_test.go.txt', HERE / 'backend_test.go.txt']
    overlay = {str(reference / 'tensor' / f'pass19_part{index}_test.go'): str(path)
        for index, path in enumerate(paths)}
    m.command('format-backend-fixture', ['gofmt', '-w', paths[-1]])
    path = m.OUT / 'backend-overlay.json'
    path.write_text(json.dumps({'Replace': overlay}, indent=2) + '\n')
    m.command('build-backend', ['go', 'test', '-c', '-overlay=' + str(path), '-o', m.OUT / 'backend.test', './tensor'], cwd=reference)
    m.command('backend-diagnostic-checks', [m.OUT / 'backend.test', '-test.run=^TestPass(16|17)', '-test.v'])
    m.command('backend-components', [m.OUT / 'backend.test', '-test.run=^$',
        '-test.bench=BenchmarkPass16(MatMul|Add)|BenchmarkPass17Pool|BenchmarkPass19Geometry',
        '-test.benchtime=150ms', '-test.count=7', '-test.benchmem'])
