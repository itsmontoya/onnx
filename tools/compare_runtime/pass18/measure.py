"""Pass 18 sequential measurements; fixed-policy comparison driver is unchanged."""
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
m.OUT = Path(os.environ.get('ONNX_PASS18_OUT', ROOT / '.cache/compare-runtime/pass18')).resolve()
CONTROL = '2b2a0c5'


def prepare():
    p.prepare()
    directory = m.OUT / 'original'
    directory.mkdir(exist_ok=True)
    revision = m.command('control-revision', ['git', 'rev-parse', CONTROL]).strip()
    names = m.command('control-root-files', ['git', 'ls-tree', '--name-only', revision]).splitlines()
    for name in names:
        if name.endswith('.go'):
            (directory / name).write_text(m.command('control-source-' + name, ['git', 'show', revision + ':' + name]))
    overlay = {str(ROOT / path.name): str(path) for path in directory.glob('*.go')}
    for path in ROOT.glob('*.go'):
        if str(path) not in overlay:
            overlay[str(path)] = ''
    overlay_path = m.OUT / 'control-overlay.json'
    overlay_path.write_text(json.dumps({'Replace': overlay}, indent=2) + '\n')
    m.command('build-control', ['go', 'test', '-c', '-overlay=' + str(overlay_path), '-o', m.OUT / 'control.test', '.'])
    environment = json.loads((m.OUT / 'environment.json').read_text())
    environment['control_commit'] = revision
    (m.OUT / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    m.command('control-benchmark', [m.OUT / 'control.test', '-test.run=^$',
        '-test.bench=^BenchmarkMNIST$', '-test.benchtime=500ms', '-test.count=3', '-test.benchmem'])
    report = m.OUT / 'fresh-baseline.json'
    m.command('fresh-baseline-run', [m.PYTHON, 'tools/compare_runtime.py', '--report', report],
        {'GOFLAGS': '-overlay=' + str(overlay_path)})
    assert json.loads(report.read_text())['passed']


def comparisons():
    for repeat in range(1, 8):
        variants = ['control', 'candidate'] if repeat % 2 else ['candidate', 'control']
        for variant in variants:
            m.bench(f'{variant}-ramp-{repeat}', variant, '^BenchmarkMNIST$')
            report_path = m.OUT / f'{variant}-comparison-{repeat}.json'
            extra = {'GOFLAGS': '-overlay=' + str(m.OUT / 'control-overlay.json')} if variant == 'control' else {}
            m.command(f'{variant}-comparison-{repeat}-run',
                [m.PYTHON, 'tools/compare_runtime.py', '--report', report_path], extra)
            report = json.loads(report_path.read_text())
            assert report['passed'] and report['timing']['go']['samples'] == 1900
            print({engine: report['timing'][engine]['median_us'] for engine in ['go', 'ort']}, flush=True)


def candidate():
    overlay = {str(ROOT / path.name): str(path) for path in (m.OUT / 'original').glob('*.go')}
    for path in ROOT.glob('*.go'):
        if path.name not in {p.name for p in (m.OUT / 'original').glob('*.go')}:
            overlay[str(path)] = ''
    (m.OUT / 'control-overlay.json').write_text(json.dumps({'Replace': overlay}, indent=2) + '\n')
    m.command('build-candidate', ['go', 'test', '-c', '-o', m.OUT / 'candidate.test', '.'])
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*sorted(ROOT.glob('*.go')), ROOT / 'go.mod', ROOT / 'go.sum']}
    (m.OUT / 'candidate-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'candidate', 'comparisons'])
    args = parser.parse_args()
    m.OUT.mkdir(parents=True, exist_ok=True)
    with (m.OUT / 'measurement.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        globals()[args.phase]()
