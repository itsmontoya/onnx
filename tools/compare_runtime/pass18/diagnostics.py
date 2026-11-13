"""Cold/amortized/workload and allocation/cost measurements, sequentially."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass18', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m

if __name__ == '__main__':
    m.command('format-benchmark-fixture', ['gofmt', '-w', HERE / 'benchmark_test.go.txt'])
    fixture = {str(p.ROOT / 'pass18_benchmark_test.go'): str(HERE / 'benchmark_test.go.txt')}
    for variant in ['control', 'candidate', 'prototype']:
        overlay = dict(fixture)
        if variant != 'candidate':
            overlay.update(json.loads((m.OUT / (variant + '-overlay.json')).read_text())['Replace'])
            overlay.update(fixture)
        path = m.OUT / (variant + '-diagnostics-overlay.json')
        path.write_text(json.dumps({'Replace': overlay}, indent=2) + '\n')
        m.command('build-' + variant + '-diagnostics', ['go', 'test', '-c', '-overlay=' + str(path),
            '-o', m.OUT / (variant + '-diagnostics.test'), '.'])
        m.bench(variant + '-amortization', variant + '-diagnostics', '^BenchmarkPass18$', '200ms', 7)
        m.bench(variant + '-cold-session', variant + '-diagnostics', '^BenchmarkPass18ColdAndSession$', '200ms', 7)
        m.bench(variant + '-workloads', variant + '-diagnostics', '^BenchmarkPass18(Workloads|DynamicShapes|Initialization)$', '200ms', 5)
        m.command(variant + '-concurrent-2', [m.OUT / (variant + '-diagnostics.test'), '-test.run=^$',
            '-test.bench=^BenchmarkPass18Workloads/ConcurrentSessions$', '-test.benchtime=300ms',
            '-test.count=5', '-test.benchmem'], {'GOMAXPROCS': '2'})
    m.command('candidate-alloc-run', [m.OUT / 'candidate.test', '-test.run=^$', '-test.bench=^BenchmarkMNIST$',
        '-test.benchtime=1000x', '-test.memprofile=' + str(m.OUT / 'candidate-mem.pprof'), '-test.memprofilerate=1'])
    for sample in ['alloc_space', 'alloc_objects']:
        m.command('candidate-' + sample, ['go', 'tool', 'pprof', '-top', '-nodecount=70',
            '-sample_index=' + sample, '-focus=onnx.*Session.*Run', m.OUT / 'candidate.test', m.OUT / 'candidate-mem.pprof'])
        m.command('candidate-' + sample + '-sites', ['go', 'tool', 'pprof', '-list=.',
            '-sample_index=' + sample, '-focus=onnx.*Session.*Run', m.OUT / 'candidate.test', m.OUT / 'candidate-mem.pprof'])
    m.bench('candidate-pipeline', 'candidate', '^BenchmarkMNISTPipeline$', '200ms', 7)
    m.bench('candidate-nodes', 'candidate', '^BenchmarkMNISTNodes/(00_Conv|01_Add|02_Relu/WithCorrection|03_MaxPool|04_Conv$|05_Add|06_Relu/WithCorrection|07_MaxPool|10_MatMul|11_Add)$', '200ms', 7)
    m.command('candidate-cpu-run', [m.OUT / 'candidate.test', '-test.run=^$', '-test.bench=^BenchmarkMNIST$',
        '-test.benchtime=5s', '-test.cpuprofile=' + str(m.OUT / 'candidate-cpu.pprof')])
    m.command('candidate-cpu-top', ['go', 'tool', 'pprof', '-top', '-nodecount=70', m.OUT / 'candidate.test', m.OUT / 'candidate-cpu.pprof'])

    m.bench('candidate-relu', 'candidate', '^BenchmarkMNISTNodes$/(02_Relu|06_Relu)$/^WithCorrection$', '200ms', 7)
    for variant in ['control', 'candidate']:
        m.bench(variant + '-load', variant, '^BenchmarkMNISTPreparation$/^LoadBytes$', '200ms', 7)
    overlay = {'Replace': {str(p.ROOT / 'pass17_fixture_test.go'):
        str(p.ROOT / 'tools/compare_runtime/pass17/fixture_test.go.txt')}}
    path = m.OUT / 'parts-overlay.json'
    path.write_text(json.dumps(overlay, indent=2) + '\n')
    m.command('build-candidate-parts', ['go', 'test', '-c', '-overlay=' + str(path), '-o', m.OUT / 'candidate-parts.test', '.'])
    m.bench('candidate-publication', 'candidate-parts', '^BenchmarkPass17Parts$/^(Publish|Clone|Arguments)$', '200ms', 7)
