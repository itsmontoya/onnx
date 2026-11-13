"""Verify final formatting, replay syntax, unchanged baseline sources and packaging."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass18', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m

if __name__ == '__main__':
    sources = subprocess.check_output(['git', 'ls-files', '*.go'], cwd=p.ROOT, text=True).splitlines()
    assert not m.command('format-final-root', ['gofmt', '-l', *sources]).strip()
    fixtures = [HERE / name for name in ['destination_storage.go.txt', 'destination_storage_test.go.txt', 'benchmark_test.go.txt']]
    assert not m.command('format-final-fixtures', ['gofmt', '-l', *fixtures]).strip()
    m.command('python-syntax-final', [sys.executable, '-m', 'py_compile', *sorted(HERE.glob('*.py'))])
    m.command('final-vet-expanded-tests', ['go', 'vet', './...'])
    m.command('diff-check-final', ['git', 'diff', '--check'])
    assert not m.command('dependency-diff-final', ['git', 'diff', '2b2a0c5', '--', 'go.mod', 'go.sum']).strip()
    for source in (m.OUT / 'original').glob('*.go'):
        expected = subprocess.check_output(['git', 'show', '2b2a0c5:' + source.name], cwd=p.ROOT)
        assert source.read_bytes() == expected, source.name
    # Final fixture formatting changes no behavior. Compile without timing new
    # observations or overwriting the previously measured diagnostic binaries.
    overlay = {'Replace': {str(p.ROOT / 'pass18_benchmark_test.go'): str(HERE / 'benchmark_test.go.txt')}}
    path = m.OUT / 'final-fixture-overlay.json'
    path.write_text(json.dumps(overlay, indent=2) + '\n')
    m.command('build-final-fixture', ['go', 'test', '-c', '-overlay=' + str(path),
        '-o', m.OUT / 'final-fixture.test', '.'])
