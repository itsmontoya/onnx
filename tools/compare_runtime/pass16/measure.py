"""Read-only backend investigation using a test-only Go overlay; no replacements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / '.cache/compare-runtime/pass16'
ENV = dict(os.environ, GOWORK='off', GOFLAGS='', GOMAXPROCS='1',
           CGO_ENABLED='0', GOARM64='v8.0',
           GOCACHE=os.environ.get('ONNX_BENCHMARK_GOCACHE', '/private/tmp/onnx-pass6/go-cache'))
PACKAGE = 'github.com/itsmontoya/neuralnetwork/tensor'
REFERENCE = OUT / 'backend-reference'


def command(name, args, extra=None, cwd=ROOT):
    started = time.perf_counter()
    result = subprocess.run([str(arg) for arg in args], cwd=cwd,
                            env=dict(ENV, **(extra or {})), capture_output=True, text=True)
    (OUT / (name + '.txt')).write_text(result.stdout + result.stderr)
    record = dict(name=name, command=[str(arg) for arg in args],
                  environment=extra or {}, cwd=str(cwd), exit_code=result.returncode,
                  elapsed_s=time.perf_counter() - started)
    with (OUT / 'commands.jsonl').open('a') as output:
        output.write(json.dumps(record) + '\n')
    print(name, result.returncode, round(record['elapsed_s'], 2), flush=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-2000:] + result.stderr[-2000:])
    return result.stdout


def prepare():
    module = json.loads(command('module', ['go', 'list', '-m', '-json',
                                          'github.com/itsmontoya/neuralnetwork']))
    assert module['Version'] == 'v0.10.9' and 'Replace' not in module
    backend = Path(module['Dir'])
    diagnostic = OUT / 'pass16_diagnostic_test.go'
    diagnostic.write_bytes((HERE / 'diagnostic_test.go.txt').read_bytes())
    command('format-diagnostic', ['gofmt', '-w', diagnostic])
    assert diagnostic.read_bytes() == (HERE / 'diagnostic_test.go.txt').read_bytes()
    # Go prohibits overlays inside GOMODCACHE. Copy the released module unchanged
    # into ONNX's ignored evidence area; neither repository nor go.mod is edited.
    if not REFERENCE.exists():
        shutil.copytree(backend, REFERENCE)
    for original in backend.rglob('*'):
        if original.is_file():
            assert original.read_bytes() == (REFERENCE / original.relative_to(backend)).read_bytes()
    overlay = {'Replace': {str(REFERENCE / 'tensor/pass16_diagnostic_test.go'): str(diagnostic)}}
    (OUT / 'overlay.json').write_text(json.dumps(overlay, indent=2) + '\n')
    hardware = json.loads(subprocess.check_output(
        ['system_profiler', 'SPHardwareDataType', '-json'], text=True))['SPHardwareDataType'][0]
    environment = dict(go=command('go-version', ['go', 'version']).strip(),
                       macos=command('macos', ['sw_vers']),
                       kernel=command('kernel', ['uname', '-a']),
                       hardware={key: hardware[key] for key in
                                 ['chip_type', 'machine_model', 'physical_memory', 'number_processors']},
                       onnx_commit=command('onnx-head', ['git', 'rev-parse', 'HEAD']).strip(),
                       module=module, performance_environment={key: ENV[key] for key in
                       ['GOWORK', 'GOFLAGS', 'GOMAXPROCS', 'CGO_ENABLED', 'GOARM64', 'GOCACHE']})
    (OUT / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    sources = [ROOT / 'go.mod', ROOT / 'go.sum', HERE / 'diagnostic_test.go.txt', HERE / 'measure.py',
               ROOT / 'mnist_diagnostic_benchmark_test.go', ROOT / 'numerical_call.go', ROOT / 'prepared_call.go',
               ROOT / '.cache/compare-runtime/mnist-ir8-opset13.onnx']
    sources += [backend / name for name in ['tensor/arithmetic.go', 'tensor/matmul.go',
                'tensor/matmul_plan.go', 'tensor/matrix_product_host.go', 'tensor/matrix_product.go',
                'matrix/matrix.go']]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (OUT / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    command('build-backend', ['go', 'test', '-c', '-overlay=' + str(OUT / 'overlay.json'),
                             '-o', OUT / 'backend.test', PACKAGE], cwd=REFERENCE)
    command('build-onnx', ['go', 'test', '-c', '-o', OUT / 'onnx.test', '.'])


def measure():
    command('diagnostic-correctness', [OUT / 'backend.test', '-test.run=^TestPass16', '-test.v'])
    command('backend-bench', [OUT / 'backend.test', '-test.run=^$', '-test.bench=^BenchmarkPass16',
                             '-test.benchtime=200ms', '-test.count=7', '-test.benchmem'])
    command('onnx-nodes', [OUT / 'onnx.test', '-test.run=^$',
                          '-test.bench=^BenchmarkMNISTNodes/(01_Add|05_Add|11_Add|10_MatMul)$',
                          '-test.benchtime=200ms', '-test.count=7', '-test.benchmem'])
    command('onnx-ramp', [OUT / 'onnx.test', '-test.run=^$', '-test.bench=^BenchmarkMNIST$',
                         '-test.benchtime=200ms', '-test.count=7', '-test.benchmem'])
    for name in ['Add', 'MatMul']:
        profile = OUT / (name.lower() + '-mem.pprof')
        command(name.lower() + '-alloc-run', [OUT / 'backend.test', '-test.run=^$',
                '-test.bench=^BenchmarkPass16' + name + '/Complete$', '-test.benchtime=10000x',
                '-test.memprofile=' + str(profile), '-test.memprofilerate=1'])
        for sample in ['alloc_objects', 'alloc_space']:
            command(name.lower() + '-' + sample, ['go', 'tool', 'pprof', '-top', '-nodecount=40',
                    '-sample_index=' + sample, '-focus=tensor\\.(AddInto|MatMulInto)',
                    OUT / 'backend.test', profile])
        expression = ('matrix.FromSlice|matrix.New|matrix.*Values' if name == 'Add'
                      else 'tensor.newMatMulPlan|tensor.matMulBatchesInto')
        command(name.lower() + '-sites', ['go', 'tool', 'pprof', '-list=' + expression,
                '-sample_index=alloc_space', '-focus=tensor\\.(AddInto|MatMulInto)',
                OUT / 'backend.test', profile])
    command('escape', ['go', 'test', '-c', '-o', OUT / 'escape.test',
                       '-gcflags=github.com/itsmontoya/neuralnetwork/tensor=-m=2',
                       '-gcflags=github.com/itsmontoya/neuralnetwork/matrix=-m=2', PACKAGE], cwd=REFERENCE)
    command('objdump', ['go', 'tool', 'objdump', '-s',
            'tensor.matrixBinaryInto|tensor.matMulBatchesInto|matrix.FromSlice|matrix.New|matrix.*Values',
            OUT / 'backend.test'])


def checks():
    overlay = '-overlay=' + str(OUT / 'overlay.json')
    for name, flags, extra in [
        ('no-cgo', [], {}), ('purego', ['-tags=purego'], {}),
        ('race', ['-race'], {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}),
        ('purego-race', ['-race', '-tags=purego'], {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}),
    ]:
        command('diagnostic-' + name, ['go', 'test', overlay, '-count=1', *flags,
                '-run=^TestPass16', PACKAGE], extra, cwd=REFERENCE)
        command('onnx-' + name, ['go', 'test', '-count=1', *flags, './...'], extra)
    command('vet', ['go', 'vet', './...'])
    command('module-verify', ['go', 'mod', 'verify'])
    command('diff-check', ['git', 'diff', '--check'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'measure', 'checks'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[args.phase]()
