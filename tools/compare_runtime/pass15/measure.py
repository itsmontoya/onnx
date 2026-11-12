"""Sequential Pass 15 measurements; the comparison driver is unchanged."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '.cache/compare-runtime/pass15'
BACKEND = Path(os.environ.get('ONNX_PASS15_BACKEND', str(ROOT.parent/'neuralnetwork')))
PYTHON = os.environ.get('ONNX_REFERENCE_PYTHON', '/private/tmp/neuralnetwork-onnx-reference/bin/python')
ENV = dict(os.environ, GOCACHE=os.environ.get('ONNX_BENCHMARK_GOCACHE', '/private/tmp/onnx-pass6/go-cache'),
           CGO_ENABLED='0', GOMAXPROCS='1', GOWORK='off', GOARM64='v8.0', GOFLAGS='',
           PYTHONPATH=os.environ.get('ONNX_REFERENCE_PYTHONPATH',
               str(ROOT.parent/'neuralnetwork/.benchmarks/conv2d-pass8/20261111-standalone/python-site')))


def command(name, args, extra=None, cwd=ROOT):
    started = time.perf_counter()
    completed = subprocess.run([str(arg) for arg in args], cwd=cwd, env=dict(ENV, **(extra or {})),
                               text=True, capture_output=True)
    (OUT/(name+'.txt')).write_text(completed.stdout+completed.stderr)
    record = dict(name=name, command=[str(arg) for arg in args], environment=extra or {}, cwd=str(cwd),
                  exit_code=completed.returncode, elapsed_s=time.perf_counter()-started)
    with (OUT/'commands.jsonl').open('a') as output:
        output.write(json.dumps(record)+'\n')
    print(name, completed.returncode, round(record['elapsed_s'], 2), flush=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout[-4000:]+completed.stderr[-4000:])
    return completed.stdout


def flags(variant):
    return '-modfile='+str(OUT/(variant+'.mod'))


def prepare():
    for variant in ['control', 'candidate']:
        for suffix in ['mod','sum']:
            payload = (subprocess.check_output(['git','show','a73fd4b:go.'+suffix],cwd=ROOT)
                       if variant == 'control' else (ROOT/('go.'+suffix)).read_bytes())
            (OUT/(variant+'.'+suffix)).write_bytes(payload)
        if variant == 'candidate':
            command('candidate-replace', ['go', 'mod', 'edit', flags(variant),
                    '-replace=github.com/itsmontoya/neuralnetwork='+str(BACKEND)])
        command('build-'+variant, ['go', 'test', '-c', flags(variant), '-o', OUT/(variant+'.test'), '.'])
        command('module-'+variant, ['go', 'list', '-m', '-json', flags(variant),
                                   'github.com/itsmontoya/neuralnetwork'])
    hardware = json.loads(subprocess.check_output(['system_profiler', 'SPHardwareDataType', '-json']))['SPHardwareDataType'][0]
    environment = dict(go=subprocess.check_output(['go', 'version'], text=True).strip(),
        macos=subprocess.check_output(['sw_vers'], text=True), kernel=subprocess.check_output(['uname', '-a'], text=True),
        hardware={key:hardware[key] for key in ['chip_type','machine_model','physical_memory','number_processors']},
        performance_environment={key:ENV[key] for key in ['CGO_ENABLED','GOMAXPROCS','GOWORK','GOARM64','GOFLAGS','GOCACHE']},
        onnx_sha=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        backend_sha=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT.parent/'neuralnetwork', text=True).strip())
    (OUT/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    sources = [ROOT/'go.mod', ROOT/'go.sum', ROOT/'tools/compare_runtime.py',
               *sorted((ROOT/'tools/compare_runtime').glob('*.go'))]
    hashes = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for name in ['arithmetic.go','broadcast.go','broadcast_add.go','destination.go']:
        hashes['candidate/tensor/'+name] = hashlib.sha256((BACKEND/'tensor'/name).read_bytes()).hexdigest()
    (OUT/'source-sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')


def bench(name, variant, pattern, duration='500ms', count=1):
    command(name, [OUT/(variant+'.test'), '-test.run=^$', '-test.bench='+pattern,
            '-test.benchtime='+duration, '-test.count='+str(count), '-test.benchmem'])


def comparisons():
    start = int(os.environ.get("ONNX_PASS15_START", "0"))
    for repeat in range(start, start+7):
        for variant in (['control','candidate'] if repeat%2 == 0 else ['candidate','control']):
            bench(f'{variant}-ramp-{repeat+1}', variant, '^BenchmarkMNIST$')
            report_path = OUT/f'{variant}-comparison-{repeat+1}.json'
            command(f'{variant}-comparison-{repeat+1}-run',
                    [PYTHON, 'tools/compare_runtime.py', '--report', report_path], {'GOFLAGS':flags(variant)})
            report = json.loads(report_path.read_text())
            assert report['passed'] and report['timing']['go']['samples'] == 1900
            print({engine:report['timing'][engine]['median_us'] for engine in ['go','ort']}, flush=True)


def components():
    for variant in ['control','candidate']:
        bench(variant+'-nodes', variant, '^BenchmarkMNISTNodes/(01_Add|05_Add|11_Add|00_Conv|02_Relu|03_MaxPool|04_Conv|06_Relu|07_MaxPool|10_MatMul)$', '200ms', 5)
        bench(variant+'-load', variant, '^BenchmarkMNISTPreparation$', '200ms', 5)
        command(variant+'-alloc-run', [OUT/(variant+'.test'), '-test.run=^$', '-test.bench=^BenchmarkMNIST$',
                '-test.benchtime=1000x', '-test.memprofile='+str(OUT/(variant+'-mem.pprof')), '-test.memprofilerate=1'])
        for sample in ['alloc_space','alloc_objects']:
            command(variant+'-'+sample, ['go','tool','pprof','-top','-nodecount=50','-sample_index='+sample,
                    '-focus=onnx.*Session.*Run',
                    OUT/(variant+'.test'), OUT/(variant+'-mem.pprof')])
        command(variant+'-objdump', ['go','tool','objdump','-s',
                'tensor.(binaryFloat32Into|broadcastAddFloat32Into|broadcastStrides|broadcastDimensions)', OUT/(variant+'.test')])
    command('candidate-cpu-run', [OUT/'candidate.test','-test.run=^$','-test.bench=^BenchmarkMNIST$',
            '-test.benchtime=5s','-test.cpuprofile='+str(OUT/'candidate-cpu.pprof')])
    command('candidate-cpu-top', ['go','tool','pprof','-top','-nodecount=45',OUT/'candidate.test',OUT/'candidate-cpu.pprof'])


def checks():
    for name, options, extra in [
        ('default', [], {'CGO_ENABLED':'1','GOMAXPROCS':'2'}), ('no-cgo', [], {}),
        ('purego', ['-tags=purego'], {}), ('race', ['-race'], {'CGO_ENABLED':'1','GOMAXPROCS':'2'}),
        ('purego-race', ['-race','-tags=purego'], {'CGO_ENABLED':'1','GOMAXPROCS':'2'})]:
        command('test-'+name, ['go','test',flags('candidate'),'-count=1',*options,'./...'], extra)
        command('backend-test-'+name, ['go','test','-count=1',*options,'./...'], extra, BACKEND)
    command('vet', ['go','vet',flags('candidate'),'./...'])
    command('backend-vet', ['go','vet','./...'], cwd=BACKEND)
    command('tidy', ['go','mod','tidy','-diff'])
    command('verify', ['go','mod','verify'])
    for target in ['darwin/arm64','darwin/amd64','linux/arm64','linux/amd64','windows/amd64','linux/386']:
        goos, goarch = target.split('/')
        for tags in ['', 'purego']:
            extra = {'GOOS':goos, 'GOARCH':goarch}
            command('build-'+goos+'-'+goarch+('-purego' if tags else ''),
                    ['go','build',flags('candidate'),'-tags='+tags,'./...'], extra)
    for package,target in [('./internal/decode','FuzzProtobufEntrypoints'),
                           ('./internal/decode','FuzzTensorDecode'),('./internal/graph','FuzzGraphReferences'),
                           ('.','FuzzShapeAttributePlanning')]:
        command('fuzz-'+target, ['go','test',flags('candidate'),package,'-run=^$',
                '-fuzz=^'+target+'$','-fuzztime=3s','-parallel=2'], {'GOMAXPROCS':'2'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare','comparisons','components','checks'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'measurement.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        globals()[args.phase]()
