"""Package all Pass 17 samples without selecting favorable repetitions."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass17', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
OUT = p.m.OUT


def stats(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values),
        cv_pct=statistics.pstdev(values) / statistics.mean(values) * 100, values=values)


def benchmarks(paths):
    groups = {}
    for path in paths:
        for name, ns, size, count in re.findall(
            r'^(Benchmark\S+)\s+\d+\s+([\d.]+) ns/op\s+(\d+) B/op\s+(\d+) allocs/op',
            path.read_text(), re.M):
            groups.setdefault(name, []).append((float(ns) / 1000, int(size), int(count)))
    return {name: dict(**stats([item[0] for item in samples]),
                      bytes=sorted({item[1] for item in samples}),
                      allocs=sorted({item[2] for item in samples})) for name, samples in groups.items()}


if __name__ == '__main__':
    hashes = json.loads((OUT / 'source-sha256.json').read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((p.ROOT / name).read_bytes()).hexdigest() == expected, name
    reports = [json.loads((OUT / f'baseline-comparison-{index}.json').read_text()) for index in range(1, 8)]
    for report in reports:
        assert report['passed'] and len(report['inputs']) == 19
        assert report['model']['sha256'] == '310384d2500a063def1ed3ad5b448baeda1edd5a570f137cb82cb11b7c25c05d'
        assert report['environment'] == reports[0]['environment']
        assert report['timing']['warmup_per_case'] == 10 and report['timing']['iterations_per_case'] == 100
        assert all(report['timing'][engine]['samples'] == 1900 for engine in ['go', 'ort'])
    baseline = {engine: stats([report['timing'][engine]['median_us'] for report in reports]) for engine in ['go', 'ort']}
    baseline['ratio'] = baseline['go']['median'] / baseline['ort']['median']
    baseline['ramp'] = benchmarks(sorted(OUT.glob('baseline-ramp-*.txt')))['BenchmarkMNIST']
    assert baseline['ramp']['bytes'] == [92826] and baseline['ramp']['allocs'] == [72]
    baseline['comparison_environment'] = reports[0]['environment']
    baseline['go_load_us'] = stats([report['timing']['go_load_ns'] / 1000 for report in reports])
    baseline['go_session_us'] = stats([report['timing']['go_session_ns'] / 1000 for report in reports])
    variants = ['control', 'no-clone', 'no-publication', 'arguments', 'destinations',
                'destinations-arguments', 'destinations-arguments-no-publication',
                'current-control', 'destinations-resolved', 'destinations-resolved-no-publication',
                'machinery-control', 'no-operators', 'destinations-no-operators']
    diagnostics = {}
    for variant in variants:
        samples = benchmarks([OUT / f'{variant}-diagnostic-{index}.txt' for index in range(1, 8)])['BenchmarkPass17Run']
        assert len(samples['values']) == 7
        diagnostics[variant] = samples
    comparisons = {}
    for variant in variants:
        control = ('current-control' if variant.startswith('destinations-resolved') else
                   'machinery-control' if 'operators' in variant else 'control')
        if variant in ['control', 'current-control', 'machinery-control']:
            continue
        first, second = diagnostics[control], diagnostics[variant]
        difference = second['median'] - first['median']
        comparisons[variant] = dict(control=control, candidate=variant,
            control_us=first['median'], candidate_us=second['median'],
            difference_of_medians_us=difference, go_pct=difference / first['median'] * 100,
            paired_median_us=statistics.median([right-left for left, right in zip(first['values'], second['values'])]),
            ort_control=None, ort_candidate=None, ort_movement=None, control_ratio=None, candidate_ratio=None)
    summary = dict(environment=json.loads((OUT / 'environment.json').read_text()), baseline=baseline,
        benchmarks={name: benchmarks([OUT / filename]) for name, filename in
                    [('nodes', 'nodes.txt'), ('pipeline', 'pipeline.txt'), ('parts', 'parts.txt'),
                     ('prepared', 'prepared-current.txt'), ('backend', 'backend-components.txt')]},
        diagnostics=diagnostics, diagnostic_comparisons=comparisons,
        roadmaps={str(ratio): dict(target_us=baseline['ort']['median'] * ratio, current_go_us=baseline['go']['median'],
            nominal_difference_us=baseline['go']['median'] - baseline['ort']['median'] * ratio) for ratio in [1.25, 1.10]},
        allocation_ledger=[
            dict(purpose=name, bytes=size, objects=count) for name, size, count in [
                ('internal destination payloads', 90336, 9), ('internal tensor headers', 864, 9),
                ('internal tensor shapes', 240, 9), ('internal identities', 9, 9),
                ('output payload', 48, 1), ('output backend tensor header', 96, 1),
                ('output shape', 16, 1), ('output identity', 1, 1), ('public Tensor wrapper', 8, 1),
                ('NamedTensor result slice', 24, 1), ('evaluation', 96, 1), ('run budget', 128, 1),
                ('value reference table', 176, 1), ('symbol map header', 48, 1),
                ('execution input slices', 160, 12), ('Conv output shapes', 64, 2), ('Pool output shapes', 64, 2),
                ('Reshape validated shape', 16, 1), ('MatMul plan', 80, 1), ('MatMul shape', 16, 1),
                ('MatMul staged output', 48, 1), ('Add Matrix headers', 144, 3), ('Add Matrix payloads', 144, 3)]] )
    assert sum(row['bytes'] for row in summary['allocation_ledger']) == 92826
    assert sum(row['objects'] for row in summary['allocation_ledger']) == 72
    ort = json.loads((OUT / 'ort-node-samples.json').read_text())
    summary['ort_nodes'] = {name: stats([statistics.mean(repeat[name]) for repeat in ort['repetitions']])
                            for name in ort['repetitions'][0]}
    commands = [json.loads(line) for line in (OUT / 'commands.jsonl').read_text().splitlines()]
    assert all(command['exit_code'] == 0 for command in commands)
    required = ['test-default', 'test-no-cgo', 'test-purego', 'test-race', 'test-purego-race',
                'onnx-metal-host-serialized', 'backend-metal-host-serialized', 'backend-metal-host-stress',
                'vet', 'verify', 'tidy']
    assert set(required) <= {command['name'] for command in commands}
    summary['metal'] = {}
    for name in ['onnx-metal-host-serialized', 'backend-metal-host-serialized', 'backend-metal-host-stress',
                 'backend-metal-device-detail']:
        log = (OUT / (name + '.txt')).read_text()
        summary['metal'][name] = dict(passed=log.count('--- PASS:'), skipped=log.count('--- SKIP:'), failed=log.count('--- FAIL:'))
        if 'host' in name:
            assert '--- FAIL:' not in log and '--- SKIP:' not in log
    assert summary['metal']['backend-metal-host-stress']['passed'] == 5
    # Keep original backend trees unmodified, including the investigation copy.
    for original in p.m.BACKEND.rglob('*'):
        if original.is_file():
            assert original.read_bytes() == (OUT / 'backend-reference' / original.relative_to(p.m.BACKEND)).read_bytes()
    diagnostic_hashes = {str(path.relative_to(p.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(HERE.glob('*')) if path.is_file()}
    backend_hashes = {str(path.relative_to(p.m.BACKEND)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for directory in ['tensor', 'matrix', 'internal/float32product']
                      for path in sorted((p.m.BACKEND / directory).glob('*.go'))}
    summary['source_sha256'] = hashes
    summary['diagnostic_sha256'] = diagnostic_hashes
    summary['backend_sha256'] = backend_hashes
    (HERE / 'data').mkdir(exist_ok=True)
    (HERE / 'data/summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (HERE / 'data/validation.json').write_text(json.dumps(commands, indent=2) + '\n')
    logs = {path.name: path.read_text() for path in sorted(OUT.glob('*.txt')) if path.name != 'hardware.txt'}
    bundle = dict(summary=summary, commands=commands, reports=reports, ort_profile=ort,
        purego_report=json.loads((OUT / 'purego-comparison.json').read_text()), logs=logs)
    with gzip.GzipFile(filename=str(HERE / 'data/evidence.json.gz'), mode='wb', mtime=0) as output:
        output.write(json.dumps(bundle, separators=(',', ':')).encode())
    print(json.dumps(dict(baseline=baseline, comparisons=comparisons, roadmaps=summary['roadmaps']), indent=2))
