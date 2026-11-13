"""Validate released v0.10.10 against released v0.10.9 without replacements."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location('pass15_measure', ROOT / 'tools/compare_runtime/pass15/measure.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.OUT = ROOT / '.cache/compare-runtime/pass16-release'
m.BACKEND = Path(subprocess.check_output(['go', 'env', 'GOMODCACHE'], text=True).strip()) / 'github.com/itsmontoya/neuralnetwork@v0.10.10'


def prepare():
    for suffix in ['mod', 'sum']:
        (m.OUT / ('control.' + suffix)).write_bytes(subprocess.check_output(
            ['git', 'show', '0803d3e:go.' + suffix], cwd=ROOT))
        (m.OUT / ('candidate.' + suffix)).write_bytes((ROOT / ('go.' + suffix)).read_bytes())
    for variant, version in [('control', 'v0.10.9'), ('candidate', 'v0.10.10')]:
        options = [m.flags('control')] if variant == 'control' else []
        module = json.loads(m.command('module-' + variant,
            ['go', 'list', '-m', '-json', *options, 'github.com/itsmontoya/neuralnetwork']))
        assert module['Version'] == version and 'Replace' not in module
        m.command('build-' + variant, ['go', 'test', '-c', *options, '-o', m.OUT / (variant + '.test'), '.'])
        m.command('build-info-' + variant, ['go', 'version', '-m', m.OUT / (variant + '.test')])
    provenance = json.loads(m.command('release-provenance',
        ['go', 'mod', 'download', '-json', 'github.com/itsmontoya/neuralnetwork@v0.10.10']))
    assert provenance['Origin']['Hash'] == '2ffa5bd9fd2034382c49d14efc732ea4d026aa09'
    hardware = json.loads(subprocess.check_output(
        ['system_profiler', 'SPHardwareDataType', '-json'], text=True))['SPHardwareDataType'][0]
    environment = dict(go=subprocess.check_output(['go', 'version'], text=True).strip(),
        macos=subprocess.check_output(['sw_vers'], text=True),
        kernel=subprocess.check_output(['uname', '-a'], text=True),
        hardware={key: hardware[key] for key in ['chip_type', 'machine_model', 'physical_memory', 'number_processors']},
        performance_environment={key: m.ENV[key] for key in ['CGO_ENABLED', 'GOMAXPROCS', 'GOWORK', 'GOARM64', 'GOFLAGS', 'GOCACHE']},
        onnx_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        publication=provenance)
    (m.OUT / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    sources = [ROOT / 'go.mod', ROOT / 'go.sum', ROOT / 'tools/compare_runtime.py',
               *sorted((ROOT / 'tools/compare_runtime').glob('*.go')),
               ROOT / '.cache/compare-runtime/mnist-ir8-opset13.onnx',
               m.BACKEND / 'tensor/arithmetic.go', m.BACKEND / 'tensor/matmul.go']
    fingerprints = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (m.OUT / 'source-sha256.json').write_text(json.dumps(fingerprints, indent=2) + '\n')


def comparisons():
    for repeat in range(7):
        for variant in (['control', 'candidate'] if repeat % 2 == 0 else ['candidate', 'control']):
            m.bench(f'{variant}-ramp-{repeat + 1}', variant, '^BenchmarkMNIST$')
            path = m.OUT / f'{variant}-comparison-{repeat + 1}.json'
            m.command(f'{variant}-comparison-{repeat + 1}-run',
                [m.PYTHON, 'tools/compare_runtime.py', '--report', path],
                {'GOFLAGS': m.flags('control') if variant == 'control' else ''})
            report = json.loads(path.read_text())
            assert report['passed'] and report['timing']['go']['samples'] == 1900
            print({engine: report['timing'][engine]['median_us'] for engine in ['go', 'ort']}, flush=True)


def components():
    for variant in ['control', 'candidate']:
        m.bench(variant + '-nodes', variant,
            '^BenchmarkMNISTNodes/(01_Add|05_Add|11_Add|10_MatMul)$', '200ms', 7)
        m.bench(variant + '-load', variant, '^BenchmarkMNISTPreparation$', '200ms', 5)
        m.command(variant + '-alloc-run', [m.OUT / (variant + '.test'), '-test.run=^$',
            '-test.bench=^BenchmarkMNIST$', '-test.benchtime=1000x',
            '-test.memprofile=' + str(m.OUT / (variant + '-mem.pprof')), '-test.memprofilerate=1'])
        for sample in ['alloc_space', 'alloc_objects']:
            m.command(variant + '-' + sample, ['go', 'tool', 'pprof', '-top', '-nodecount=50',
                '-sample_index=' + sample, '-focus=onnx.*Session.*Run',
                m.OUT / (variant + '.test'), m.OUT / (variant + '-mem.pprof')])


def checks():
    m.checks()
    m.command('test-metal', ['go', 'test', '-count=1', '-tags=metal', './...'],
        {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'})
    m.command('backend-test-metal', ['go', 'test', '-count=1', '-tags=metal', './...'],
        {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}, m.BACKEND)
    m.command('backend-metal-detail', ['go', 'test', '-count=1', '-v', '-tags=metal', './matrix', './internal/device'],
        {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}, m.BACKEND)
    m.command('candidate-purego-comparison-run',
        [m.PYTHON, 'tools/compare_runtime.py', '--report', m.OUT / 'candidate-purego.json'],
        {'GOFLAGS': '-tags=purego'})
    paths = subprocess.check_output(['git', 'ls-files', '*.go'], cwd=ROOT, text=True).splitlines()
    assert not m.command('format', ['gofmt', '-l', *paths]).strip()
    m.command('diff-check', ['git', 'diff', '--check'])


def metal_host():
    m.command('test-metal-host', ['go', 'test', '-count=1', '-tags=metal', './...'],
        {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'})
    m.command('backend-test-metal-host', ['go', 'test', '-count=1', '-tags=metal', './...'],
        {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}, m.BACKEND)
    m.command('backend-metal-host-detail', ['go', 'test', '-count=1', '-v', '-tags=metal', './matrix', './internal/device'],
        {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}, m.BACKEND)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'comparisons', 'components', 'checks', 'metal-host'])
    args = parser.parse_args()
    m.OUT.mkdir(parents=True, exist_ok=True)
    with (m.OUT / 'measurement.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        globals()[args.phase.replace('-', '_')]()
