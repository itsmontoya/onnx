"""Run isolated backend, dynamic fallback, compiler and machinery diagnostics."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('measure', HERE/'measure.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
BACKEND = m.BACKEND
ARTIFACTS = BACKEND/'.benchmarks/add-pass15/run'
ARTIFACTS.mkdir(parents=True, exist_ok=True)
# The control must remain the released source after the sibling checkout changes.
control = Path(json.loads((m.OUT/'module-control.txt').read_text())['Dir'])
empty = ARTIFACTS/'empty_test.go'
empty.write_text('package tensor\n')
replacements = {str(BACKEND/'tensor'/name):str(control/'tensor'/name)
                for name in ['arithmetic.go','broadcast.go','broadcast_add.go','destination.go']}
replacements[str(BACKEND/'tensor/broadcast_test.go')] = str(empty)
(ARTIFACTS/'control-overlay.json').write_text(json.dumps({'Replace':replacements}))


for variant in ['control','candidate']:
    options = ['-overlay='+str(ARTIFACTS/'control-overlay.json')] if variant == 'control' else []
    binary = ARTIFACTS/(variant+'-dynamic.test')
    m.command('backend-build-'+variant, ['go','test','-c',*options,'-o',binary,'./tensor'], cwd=BACKEND)
    m.command(variant+'-dynamic', [binary,'-test.run=^$',
        '-test.bench=^BenchmarkAddInto/(MNISTFirst|MNISTSecond|Scalar|Batch|RankFive|Fallback|FallbackVector|RankEight|RankNine|Equal|Empty)$',
        '-test.benchtime=100ms','-test.count=5','-test.benchmem'], cwd=BACKEND)
    m.bench(variant+'-dynamic-onnx', variant, '^BenchmarkPreparedCall$', '100ms', 5)
    for sample in ['alloc_space','alloc_objects']:
        m.command(variant+'-run-'+sample, ['go','tool','pprof','-top','-nodefraction=0','-nodecount=60',
            '-sample_index='+sample,'-focus=onnx.*Session.*Run',m.OUT/(variant+'.test'),m.OUT/(variant+'-mem.pprof')])
m.bench('candidate-machinery','candidate','^BenchmarkMNISTPipeline/Bind$', '200ms', 5)
m.command('candidate-escape',['go','test','-c','-gcflags=github.com/itsmontoya/neuralnetwork/tensor=-m=2',
    '-o',ARTIFACTS/'escape.test','./tensor'], cwd=BACKEND)
m.command('final-backend-default',['go','test','-count=1','./...'], cwd=BACKEND)
m.command('final-onnx-default',['go','test',m.flags('candidate'),'-count=1','./...'])
m.command('ort-nodes',[m.PYTHON,HERE/'ort_profile.py'])
