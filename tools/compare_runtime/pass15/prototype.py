"""Diagnostic execution-plan-owned immutable Add descriptors, never production."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('measure', HERE/'measure.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

replacements = {}
for source, target in [
    ('prepared_backend.go', m.BACKEND/'tensor/pass15_add_plan.go'),
    ('prepared_numerical_call.go', m.ROOT/'numerical_call.go'),
    ('prepared_execution_plan.go', m.ROOT/'execution_plan.go'),
    ('prepared_benchmark_test.go', m.ROOT/'pass15_prepared_benchmark_test.go')]:
    copied = m.OUT/source
    copied.write_bytes((HERE/(source+'.txt')).read_bytes())
    replacements[str(target)] = str(copied)
(m.OUT/'prepared-overlay.json').write_text(json.dumps({'Replace':replacements}))
overlay = '-overlay='+str(m.OUT/'prepared-overlay.json')
proto_flags = m.flags('candidate')+' '+overlay
m.command('prepared-build', ['go','test',m.flags('candidate'),overlay,'-c','-o',m.OUT/'prepared.test','.'])
m.command('prepared-tests', ['go','test',m.flags('candidate'),overlay,'-count=1','.'])
m.bench('prepared-components','prepared','^BenchmarkPass15Descriptor(Init)?$', '200ms',5)
m.command('prepared-ramp',[m.OUT/'prepared.test','-test.run=^$','-test.bench=^BenchmarkMNIST$',
    '-test.benchtime=300ms','-test.count=5','-test.benchmem'])
for repeat in range(7):
    for variant in (['ownership','prepared'] if repeat%2 == 0 else ['prepared','ownership']):
        report = m.OUT/f'prototype-{variant}-{repeat+1}.json'
        flags = proto_flags if variant == 'prepared' else m.flags('candidate')
        m.command(f'prototype-{variant}-{repeat+1}',[m.PYTHON,'tools/compare_runtime.py','--report',report],{'GOFLAGS':flags})
        assert json.loads(report.read_text())['passed']
