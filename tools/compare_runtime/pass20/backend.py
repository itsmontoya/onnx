"""Time published geometry helpers with test-only overlays on an exact copy."""
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass20', HERE / 'measure.py')
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
        HERE.parent / 'pass17/backend_test.go.txt', HERE.parent / 'pass19/backend_test.go.txt']
    overlay = {str(reference / 'tensor' / f'pass20_geometry{index}_test.go'): str(path)
        for index, path in enumerate(paths)}
    path = m.OUT / 'backend-overlay.json'
    path.write_text(json.dumps({'Replace': overlay}, indent=2) + '\n')
    m.command('build-backend', ['go', 'test', '-c', '-overlay=' + str(path),
        '-o', m.OUT / 'backend.test', './tensor'], cwd=reference)
    m.command('backend-geometry', [m.OUT / 'backend.test', '-test.run=^$',
        '-test.bench=^BenchmarkPass19Geometry$|^BenchmarkPass17Pool$/./^(Validation|Shape)$',
        '-test.benchtime=150ms', '-test.count=7', '-test.benchmem'])
