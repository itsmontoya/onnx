"""Sequential Pass 17 investigation of the unchanged published runtime."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('measurement', HERE.parent / 'pass15/measure.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.OUT = ROOT / '.cache/compare-runtime/pass17'
m.BACKEND = Path(subprocess.check_output(['go', 'env', 'GOMODCACHE'], text=True).strip()) / 'github.com/itsmontoya/neuralnetwork@v0.10.10'


def prepare():
    module = json.loads(m.command('module', ['go', 'list', '-m', '-json', 'github.com/itsmontoya/neuralnetwork']))
    assert module['Version'] == 'v0.10.10' and 'Replace' not in module
    publication = json.loads(m.command('publication', ['go', 'mod', 'download', '-json', 'github.com/itsmontoya/neuralnetwork@v0.10.10']))
    hardware = json.loads(m.command('hardware', ['system_profiler', 'SPHardwareDataType', '-json']))['SPHardwareDataType'][0]
    environment = dict(go=m.command('go-version', ['go', 'version']).strip(),
        macos=m.command('macos', ['sw_vers']), kernel=m.command('kernel', ['uname', '-a']),
        hardware={key: hardware[key] for key in ['chip_type', 'machine_model', 'physical_memory', 'number_processors']},
        performance_environment={key: m.ENV[key] for key in ['CGO_ENABLED', 'GOMAXPROCS', 'GOWORK', 'GOARM64', 'GOFLAGS', 'GOCACHE', 'PYTHONPATH']},
        onnx_commit=m.command('onnx-head', ['git', 'rev-parse', 'HEAD']).strip(), publication=publication,
        modules=m.command('modules', ['go', 'list', '-m', 'all']))
    (m.OUT / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    sources = [*sorted(ROOT.glob('*.go')), ROOT / 'go.mod', ROOT / 'go.sum',
        ROOT / 'tools/compare_runtime.py', *sorted((ROOT / 'tools/compare_runtime').glob('*.go')),
        ROOT / '.cache/compare-runtime/mnist-ir8-opset13.onnx']
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (m.OUT / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    m.command('build', ['go', 'test', '-c', '-o', m.OUT / 'baseline.test', '.'])
    m.command('build-info', ['go', 'version', '-m', m.OUT / 'baseline.test'])


def baseline():
    for repeat in range(1, 8):
        m.bench(f'baseline-ramp-{repeat}', 'baseline', '^BenchmarkMNIST$')
        path = m.OUT / f'baseline-comparison-{repeat}.json'
        m.command(f'baseline-comparison-{repeat}-run', [m.PYTHON, 'tools/compare_runtime.py', '--report', path])
        report = json.loads(path.read_text())
        assert report['passed'] and report['timing']['go']['samples'] == 1900
        print({engine: report['timing'][engine]['median_us'] for engine in ['go', 'ort']}, flush=True)


def profiles():
    m.command('alloc-run', [m.OUT / 'baseline.test', '-test.run=^$', '-test.bench=^BenchmarkMNIST$',
        '-test.benchtime=1000x', '-test.memprofile=' + str(m.OUT / 'mem.pprof'), '-test.memprofilerate=1'])
    for sample in ['alloc_space', 'alloc_objects']:
        m.command(sample, ['go', 'tool', 'pprof', '-top', '-nodecount=100',
            '-sample_index=' + sample, '-focus=onnx.*Session.*Run', m.OUT / 'baseline.test', m.OUT / 'mem.pprof'])
        m.command(sample + '-sites', ['go', 'tool', 'pprof', '-list=.', '-sample_index=' + sample,
            '-focus=onnx.*Session.*Run', m.OUT / 'baseline.test', m.OUT / 'mem.pprof'])
    m.bench('nodes', 'baseline', '^BenchmarkMNISTNodes/(00_Conv|01_Add|02_Relu/WithCorrection|03_MaxPool|04_Conv$|05_Add|06_Relu/WithCorrection|07_MaxPool|10_MatMul|11_Add)$', '200ms', 7)
    m.bench('pipeline', 'baseline', '^BenchmarkMNISTPipeline$', '200ms', 7)
    m.command('cpu-run', [m.OUT / 'baseline.test', '-test.run=^$', '-test.bench=^BenchmarkMNIST$',
        '-test.benchtime=5s', '-test.cpuprofile=' + str(m.OUT / 'cpu.pprof')])
    m.command('cpu-top', ['go', 'tool', 'pprof', '-top', '-nodecount=70', m.OUT / 'baseline.test', m.OUT / 'cpu.pprof'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'baseline', 'profiles'])
    args = parser.parse_args()
    m.OUT.mkdir(parents=True, exist_ok=True)
    with (m.OUT / 'measurement.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        globals()[args.phase]()
