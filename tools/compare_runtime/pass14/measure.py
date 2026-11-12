"""Sequential measurements and checks for the released v0.10.8 integration."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / '.cache/compare-runtime/pass14-release'
PYTHON = Path(os.environ.get('ONNX_REFERENCE_PYTHON','/private/tmp/neuralnetwork-onnx-reference/bin/python'))
ENV = dict(os.environ, GOCACHE=os.environ.get('ONNX_BENCHMARK_GOCACHE','/private/tmp/onnx-pass6/go-cache'), GOWORK='off',
           GOMAXPROCS='1', CGO_ENABLED='0', GOARM64='v8.0', GOFLAGS='',
           PYTHONPATH=os.environ.get('ONNX_REFERENCE_PYTHONPATH',str(ROOT.parent/'neuralnetwork/.benchmarks/conv2d-pass8/20261111-standalone/python-site')))


def command(name, args, extra=None, cwd=ROOT):
    started = time.perf_counter()
    completed = subprocess.run(args, cwd=cwd, env=dict(ENV, **(extra or {})),
                               text=True, capture_output=True)
    (OUT/(name+'.txt')).write_text(completed.stdout+completed.stderr)
    record = dict(name=name, command=[str(arg) for arg in args], environment=extra or {},
                  cwd=str(cwd), exit_code=completed.returncode,
                  elapsed_s=time.perf_counter()-started)
    with (OUT/'commands.jsonl').open('a') as output:
        output.write(json.dumps(record)+'\n')
    print(name, completed.returncode, round(record['elapsed_s'], 2), flush=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:]+completed.stderr[-5000:])
    return completed.stdout


def bench(name, variant, pattern, duration='500ms', count=1):
    command(name, [str(OUT/(variant+'.test')), '-test.run=^$', '-test.bench='+pattern,
                   '-test.benchtime='+duration, '-test.count='+str(count), '-test.benchmem'])


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT/'baseline.mod').exists():
        (OUT/'baseline.mod').write_bytes(subprocess.check_output(['git','show','47271fa:go.mod'],cwd=ROOT))
        (OUT/'baseline.sum').write_bytes(subprocess.check_output(['git','show','47271fa:go.sum'],cwd=ROOT))
    for variant in ['control','candidate']:
        flags = ['-modfile='+str(OUT/'baseline.mod')] if variant=='control' else []
        command('build-'+variant, ['go','test','-c',*flags,'-o',str(OUT/(variant+'.test')),'.'])
        command('module-'+variant, ['go','list','-m','-json',*flags,'github.com/itsmontoya/neuralnetwork'])
    hashes = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [ROOT/'go.mod',ROOT/'go.sum',ROOT/'tools/compare_runtime.py',
                           *sorted((ROOT/'tools/compare_runtime').glob('*.go'))]}
    (OUT/'source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    profile = json.loads(subprocess.check_output(['system_profiler','SPHardwareDataType','-json'],
                                                stderr=subprocess.DEVNULL))['SPHardwareDataType'][0]
    environment = dict(go=subprocess.check_output(['go','version'],text=True).strip(),
        macos=subprocess.check_output(['sw_vers'],text=True),kernel=subprocess.check_output(['uname','-a'],text=True),
        hardware={key:value for key,value in profile.items()
                  if key in ['chip_type','machine_model','physical_memory','number_processors']},
        performance_environment={key:ENV[key] for key in ['CGO_ENABLED','GOMAXPROCS','GOWORK','GOARM64','GOFLAGS','GOCACHE']})
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')


def comparisons():
    for repeat in range(7):
        for variant in (['control','candidate'] if repeat%2==0 else ['candidate','control']):
            bench(f'{variant}-ramp-{repeat+1}',variant,'^BenchmarkMNIST$')
            flags = '-modfile='+str(OUT/'baseline.mod') if variant=='control' else ''
            report_path = OUT/f'{variant}-comparison-{repeat+1}.json'
            command(f'{variant}-comparison-{repeat+1}-run',
                    [str(PYTHON),'tools/compare_runtime.py','--report',str(report_path)],{'GOFLAGS':flags})
            report = json.loads(report_path.read_text())
            assert report['passed'] and report['timing']['go']['samples']==1900
            print({engine:report['timing'][engine]['median_us'] for engine in ['go','ort']},flush=True)


def components():
    for variant in ['control','candidate']:
        bench(variant+'-nodes',variant,'^BenchmarkMNISTNodes$',duration='150ms',count=5)
    command('candidate-cpu-run',[str(OUT/'candidate.test'),'-test.run=^$','-test.bench=^BenchmarkMNIST$',
            '-test.benchtime=5s','-test.cpuprofile='+str(OUT/'candidate-cpu.pprof')])
    command('candidate-cpu-top',['go','tool','pprof','-top','-nodecount=35',
            str(OUT/'candidate.test'),str(OUT/'candidate-cpu.pprof')])
    command('candidate-alloc-run',[str(OUT/'candidate.test'),'-test.run=^$','-test.bench=^BenchmarkMNIST$',
            '-test.benchtime=1000x','-test.memprofile='+str(OUT/'candidate-mem.pprof'),'-test.memprofilerate=1'])
    for sample in ['alloc_space','alloc_objects']:
        command('candidate-'+sample,['go','tool','pprof','-top','-nodecount=35','-sample_index='+sample,
                str(OUT/'candidate.test'),str(OUT/'candidate-mem.pprof')])
    for variant in ['control','candidate']:
        command(variant+'-relu-objdump',['go','tool','objdump','-s',
                'tensor.(ReLUPreservingNaNInto|reluPreservingNaNPrefix|reluPreservingNaNBlocks|ReLUInto)',
                str(OUT/(variant+'.test'))])


def checks():
    for name, flags, extra in [
        ('test-default',[],{'CGO_ENABLED':'1','GOMAXPROCS':'2'}),
        ('test-no-cgo',[],{}),('test-purego',['-tags=purego'],{}),
        ('test-race',['-race'],{'CGO_ENABLED':'1','GOMAXPROCS':'2'}),
        ('test-purego-race',['-race','-tags=purego'],{'CGO_ENABLED':'1','GOMAXPROCS':'2'})]:
        command(name,['go','test','-count=1',*flags,'./...'],extra)
    command('test-released-backend',['go','test','-count=1','github.com/itsmontoya/neuralnetwork/...'])
    command('test-released-backend-purego',['go','test','-count=1','-tags=purego',
                                          'github.com/itsmontoya/neuralnetwork/...'])
    command('vet',['go','vet','./...'])
    command('tidy',['go','mod','tidy','-diff'])
    command('verify',['go','mod','verify'])
    command('release-provenance',['go','mod','download','-json','github.com/itsmontoya/neuralnetwork@v0.10.8'])
    paths = subprocess.check_output(['git','ls-files','*.go'],cwd=ROOT,text=True).splitlines()
    assert not command('format',['gofmt','-l',*paths]).strip()
    for target in ['darwin/arm64','darwin/amd64','linux/arm64','linux/amd64','windows/amd64']:
        goos, goarch = target.split('/')
        for tags in ['', 'purego']:
            command('build-'+goos+'-'+goarch+('-purego' if tags else ''),
                    ['go','build','-tags='+tags,'./...'],{'GOOS':goos,'GOARCH':goarch})
    for package,target in [('./internal/decode','FuzzProtobufEntrypoints'),
                           ('./internal/decode','FuzzTensorDecode'),('./internal/graph','FuzzGraphReferences'),
                           ('.','FuzzShapeAttributePlanning')]:
        command('fuzz-'+target,['go','test',package,'-run=^$','-fuzz=^'+target+'$',
                               '-fuzztime=3s','-parallel=2'],{'GOMAXPROCS':'2'})
    command('diff-check',['git','diff','--check'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','comparisons','components','checks'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'measurement.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        globals()[args.phase]()
