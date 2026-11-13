"""Check formatting, replay syntax, scope and the measured production snapshot."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass19', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m

if __name__ == '__main__':
    paths = subprocess.check_output(['git', 'ls-files', '*.go'], cwd=p.ROOT, text=True).splitlines()
    paths += [str(path) for path in HERE.glob('*.go.txt')]
    assert not m.command('format-final', ['gofmt', '-l', *paths]).strip()
    m.command('python-syntax', [sys.executable, '-m', 'py_compile', *sorted(HERE.glob('*.py'))])
    m.command('diff-check-final', ['git', 'diff', '--check'])
    changes = m.command('changed-files-final', ['git', 'diff', '--name-only']).splitlines()
    assert all(name in ['.gitignore', 'evaluation.go', 'evaluation_test.go'] or name.endswith('.md')
        or name.startswith('tools/compare_runtime/pass19/') for name in changes)
    source_changes = m.command('production-scope', ['git', 'diff', '--name-only', p.BASELINE, '--', '*.go']).splitlines()
    assert source_changes == ['evaluation.go', 'evaluation_test.go']
    m.command('fmt', ['go', 'fmt', './...'])
    for variant in ['merged', 'candidate']:
        m.command('final-build-info-' + variant, ['go', 'version', '-m', m.OUT / (variant + '.test')])
    m.command('production-commit', ['git', 'log', '-1', '--format=%H', '--', 'evaluation.go'])
    # Never reinterpret retained artifacts from another directory as this run.
    assert json.loads((m.OUT / 'environment.json').read_text())['onnx_commit'] == p.BASELINE
