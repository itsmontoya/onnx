"""Sequential Pass 20 investigation; published backend sources remain untouched."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('pass17', HERE.parent / 'pass17/measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m
m.OUT = Path(os.environ.get('ONNX_PASS20_OUT', ROOT / '.cache/compare-runtime/pass20')).resolve()
BASELINE = '06ae89fbaba9f46c0e111fc7ef3b896eccb8e899'


def prepare():
    assert not (m.OUT / 'environment.json').exists(), 'Use a fresh output directory.'
    p.prepare()
    environment = json.loads((m.OUT / 'environment.json').read_text())
    environment['working_tree_commit'] = environment['onnx_commit']
    environment['onnx_commit'] = BASELINE
    original = m.OUT / 'merged-evaluation.go'
    original.write_text(m.command('merged-source', ['git', 'show', BASELINE + ':evaluation.go']))
    overlay = m.OUT / 'merged-overlay.json'
    overlay.write_text(json.dumps({'Replace': {str(ROOT / 'evaluation.go'): str(original)}}, indent=2) + '\n')
    m.command('build-merged-baseline', ['go', 'test', '-c', '-overlay=' + str(overlay),
        '-o', m.OUT / 'baseline.test', '.'])
    hashes = json.loads((m.OUT / 'source-sha256.json').read_text())
    hashes['evaluation.go'] = hashlib.sha256(original.read_bytes()).hexdigest()
    (m.OUT / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    environment['branch'] = m.command('branch', ['git', 'branch', '--show-current']).strip()
    environment['python'] = m.command('python-environment', [m.PYTHON, '-c',
        'import sys,numpy,onnx,onnxruntime; print(sys.version); '
        'print(dict(numpy=numpy.__version__,onnx=onnx.__version__,ort=onnxruntime.__version__))'])
    (m.OUT / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    backend = {str(path.relative_to(m.BACKEND)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(m.BACKEND.rglob('*.go'))}
    (m.OUT / 'backend-sha256.json').write_text(json.dumps(backend, indent=2) + '\n')


def baseline():
    p.baseline()


def profiles():
    p.profiles()


def production():
    original = m.OUT / 'merged-evaluation.go'
    original.write_text(m.command('merged-source', ['git', 'show', BASELINE + ':evaluation.go']))
    overlay = m.OUT / 'merged-overlay.json'
    overlay.write_text(json.dumps({'Replace': {str(ROOT / 'evaluation.go'): str(original)}}, indent=2) + '\n')
    for variant, options in [('merged', ['-overlay=' + str(overlay)]), ('candidate', [])]:
        m.command('build-production-' + variant, ['go', 'test', '-c', *options,
            '-o', m.OUT / (variant + '.test'), '.'])
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*sorted(ROOT.glob('*.go')), ROOT / 'go.mod', ROOT / 'go.sum']}
    (m.OUT / 'candidate-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    for repeat in range(1, 8):
        for variant in ['merged', 'candidate'] if repeat % 2 else ['candidate', 'merged']:
            m.bench(f'production-{variant}-ramp-{repeat}', variant, '^BenchmarkMNIST$')
            path = m.OUT / f'production-{variant}-comparison-{repeat}.json'
            extra = {'GOFLAGS': '-overlay=' + str(overlay)} if variant == 'merged' else {}
            m.command(f'production-{variant}-comparison-{repeat}-run',
                [m.PYTHON, 'tools/compare_runtime.py', '--report', path], extra)
            report = json.loads(path.read_text())
            assert report['passed'] and report['timing']['go']['samples'] == 1900
            print({engine: report['timing'][engine]['median_us'] for engine in ['go', 'ort']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'baseline', 'profiles', 'production'])
    args = parser.parse_args()
    m.OUT.mkdir(parents=True, exist_ok=True)
    with (m.OUT / 'measurement.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        globals()[args.phase]()
