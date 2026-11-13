"""Record generated code, exact shape comparison and final allocations."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('diagnostics', HERE / 'diagnostics.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
m = d.m

if __name__ == '__main__':
    d.build('control')
    m.bench('metadata-compare-exact', 'control', '^BenchmarkPass20Metadata$/^CompareExact$', '150ms', 7)
    for name, options in [('merged', ['-overlay=' + str(m.OUT / 'merged-overlay.json')]), ('candidate', [])]:
        m.command(name + '-escape', ['go', 'test', '-c', *options,
            '-gcflags=github.com/itsmontoya/onnx=-m=2', '-o', m.OUT / (name + '-escape.test'), '.'])
        m.command(name + '-objdump', ['go', 'tool', 'objdump', '-s',
            r'onnx.\(\*evaluation\).checkValue|onnx.\(\*destinationStorage\).get', m.OUT / (name + '-escape.test')])
    m.command('candidate-alloc-run', [m.OUT / 'candidate.test', '-test.run=^$', '-test.bench=^BenchmarkMNIST$',
        '-test.benchtime=1000x', '-test.memprofile=' + str(m.OUT / 'candidate-mem.pprof'), '-test.memprofilerate=1'])
    m.command('candidate-alloc-sites', ['go', 'tool', 'pprof', '-list=.', '-sample_index=alloc_objects',
        '-focus=onnx.*Session.*Run', m.OUT / 'candidate.test', m.OUT / 'candidate-mem.pprof'])
