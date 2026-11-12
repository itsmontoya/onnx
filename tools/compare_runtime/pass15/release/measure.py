"""Validate the published v0.10.9 integration without replacing either module."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass15', HERE.parent/'measure.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.OUT = m.ROOT/'.cache/compare-runtime/pass15-release'
m.BACKEND = Path(subprocess.check_output(['go','env','GOMODCACHE'], text=True).strip())/'github.com/itsmontoya/neuralnetwork@v0.10.9'


def prepare():
    if not (m.OUT/'control.mod').exists():
        for suffix in ['mod','sum']:
            (m.OUT/('control.'+suffix)).write_bytes(subprocess.check_output(
                ['git','show','a73fd4b:go.'+suffix], cwd=m.ROOT))
    for suffix in ['mod','sum']:
        (m.OUT/('candidate.'+suffix)).write_bytes((m.ROOT/('go.'+suffix)).read_bytes())
    for variant in ['control','candidate']:
        options = [m.flags('control')] if variant == 'control' else []
        m.command('build-'+variant, ['go','test','-c',*options,'-o',m.OUT/(variant+'.test'),'.'])
        module = json.loads(m.command('module-'+variant,
            ['go','list','-m','-json',*options,'github.com/itsmontoya/neuralnetwork']))
        assert 'Replace' not in module
        assert module['Version'] == ('v0.10.8' if variant == 'control' else 'v0.10.9')
    provenance = json.loads(m.command('release-provenance',
        ['go','mod','download','-json','github.com/itsmontoya/neuralnetwork@v0.10.9']))
    (m.OUT/'release-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    hardware = json.loads(subprocess.check_output(['system_profiler','SPHardwareDataType','-json']))['SPHardwareDataType'][0]
    environment = dict(go=subprocess.check_output(['go','version'],text=True).strip(),
        macos=subprocess.check_output(['sw_vers'],text=True),kernel=subprocess.check_output(['uname','-a'],text=True),
        hardware={key:hardware[key] for key in ['chip_type','machine_model','physical_memory','number_processors']},
        performance_environment={key:m.ENV[key] for key in ['CGO_ENABLED','GOMAXPROCS','GOWORK','GOARM64','GOFLAGS','GOCACHE']},
        onnx_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=m.ROOT,text=True).strip(),
        release_sha=provenance['Origin']['Hash'])
    (m.OUT/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    expected = json.loads((HERE.parent/'data/summary.json').read_text())['source_sha256']
    hashes = {}
    for name in ['arithmetic.go','broadcast.go','broadcast_add.go','destination.go']:
        hashes[name] = hashlib.sha256((m.BACKEND/'tensor'/name).read_bytes()).hexdigest()
        assert hashes[name] == expected['candidate/tensor/'+name]
    (m.OUT/'source-identity.json').write_text(json.dumps(hashes,indent=2)+'\n')


def comparisons():
    for repeat in range(7):
        for variant in (['control','candidate'] if repeat%2 == 0 else ['candidate','control']):
            m.bench(f'{variant}-ramp-{repeat+1}',variant,'^BenchmarkMNIST$')
            path = m.OUT/f'{variant}-comparison-{repeat+1}.json'
            m.command(f'{variant}-comparison-{repeat+1}-run',
                [m.PYTHON,'tools/compare_runtime.py','--report',path],
                {'GOFLAGS':m.flags('control') if variant == 'control' else ''})
            report = json.loads(path.read_text())
            assert report['passed'] and report['timing']['go']['samples'] == 1900
            print({engine:report['timing'][engine]['median_us'] for engine in ['go','ort']},flush=True)


def checks():
    m.checks()
    paths = subprocess.check_output(['git','ls-files','*.go'],cwd=m.ROOT,text=True).splitlines()
    assert not m.command('format',['gofmt','-l',*paths]).strip()
    m.command('diff-check',['git','diff','--check'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','comparisons','components','checks'])
    args = parser.parse_args()
    m.OUT.mkdir(parents=True,exist_ok=True)
    with (m.OUT/'measurement.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.phase == 'components':
            m.components()
            m.command('ort-nodes',[m.PYTHON,HERE/'ort_profile.py'])
        else:
            globals()[args.phase]()
