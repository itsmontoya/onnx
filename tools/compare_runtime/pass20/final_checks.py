"""Check measured source identity, formatting, tooling syntax and change scope."""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass20', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m

if __name__ == '__main__':
    paths = m.command('tracked-go-files', ['git', 'ls-files', '*.go']).splitlines()
    paths += [str(path) for path in HERE.glob('*.go.txt')]
    assert not m.command('format-final', ['gofmt', '-l', *paths]).strip()
    m.command('python-syntax', [sys.executable, '-m', 'py_compile', *sorted(HERE.glob('*.py'))])
    for name in ['measure', 'diagnostics', 'checks']:
        m.command(name + '-help', [sys.executable, HERE / (name + '.py'), '--help'])
    m.command('diff-check-final', ['git', 'diff', '--check'])
    changes = m.command('production-scope', ['git', 'diff', '--name-only', p.BASELINE, '--', '*.go']).splitlines()
    assert changes == ['evaluation.go', 'evaluation_test.go'], changes
    m.command('fmt', ['go', 'fmt', './...'])
    for variant in ['merged', 'candidate']:
        m.command('final-build-info-' + variant, ['go', 'version', '-m', m.OUT / (variant + '.test')])
    assert json.loads((m.OUT / 'environment.json').read_text())['onnx_commit'] == p.BASELINE
