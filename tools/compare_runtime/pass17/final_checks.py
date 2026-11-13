"""Verify evidence sources, formatting, and replay syntax before packaging."""
import importlib.util
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass17', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m

if __name__ == '__main__':
    paths = subprocess.check_output(['git', 'ls-files', '*.go'], cwd=p.ROOT, text=True).splitlines()
    paths += [str(HERE / name) for name in ['fixture_test.go.txt', 'backend_test.go.txt']]
    assert not m.command('format-final', ['gofmt', '-l', *paths]).strip()
    m.command('python-syntax', [sys.executable, '-m', 'py_compile', *sorted(HERE.glob('*.py'))])
    m.command('diff-check-final', ['git', 'diff', '--check'])
    changed = m.command('changed-files-final', ['git', 'diff', '--name-only']).splitlines()
    assert all(name == '.gitignore' or name.endswith('.md') or name.startswith('tools/compare_runtime/pass17/') for name in changed)
    m.command('alloc_objects-correct-units', ['go', 'tool', 'pprof', '-top', '-nodecount=100',
        '-sample_index=alloc_objects', '-focus=onnx.*Session.*Run', m.OUT / 'baseline.test', m.OUT / 'mem.pprof'])
