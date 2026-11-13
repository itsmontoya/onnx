"""Package every valid Pass 18 observation and recorded failure without filtering."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import statistics as s

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = Path(os.environ.get('ONNX_PASS18_OUT', ROOT / '.cache/compare-runtime/pass18')).resolve()
DATA = HERE / 'data'


def stats(values):
    return dict(median=s.median(values), minimum=min(values), maximum=max(values),
        cv_pct=100*s.pstdev(values)/s.mean(values), values=values)


def benchmarks(path):
    rows = {}
    for name, elapsed, allocated, count in re.findall(
            r'^(Benchmark\S+)\s+\d+\s+([\d.]+) ns/op\s+(\d+) B/op\s+(\d+) allocs/op',
            path.read_text(), re.M):
        row = rows.setdefault(name, dict(latency_us=[], bytes=[], allocs=[]))
        row['latency_us'].append(float(elapsed)/1000)
        row['bytes'].append(int(allocated))
        row['allocs'].append(int(count))
    return {name: dict(latency_us=stats(row['latency_us']), bytes=row['bytes'], allocs=row['allocs'])
        for name, row in rows.items()}


if __name__ == '__main__':
    DATA.mkdir(exist_ok=True)
    reports = {variant: [json.loads((OUT / f'{variant}-comparison-{repeat}.json').read_text())
        for repeat in range(1, 8)] for variant in ['control', 'candidate']}
    fixed = {variant: {engine: stats([r['timing'][engine]['median_us'] for r in samples])
        for engine in ['go', 'ort']} for variant, samples in reports.items()}
    for variant in fixed:
        fixed[variant]['ratio'] = fixed[variant]['go']['median']/fixed[variant]['ort']['median']
    go_difference = fixed['control']['go']['median'] - fixed['candidate']['go']['median']
    ort_movement = fixed['candidate']['ort']['median'] - fixed['control']['ort']['median']
    paired = [left['timing']['go']['median_us'] - right['timing']['go']['median_us']
        for left, right in zip(reports['control'], reports['candidate'])]
    ramp = {}
    for variant in ['control', 'candidate']:
        rows = [benchmarks(OUT / f'{variant}-ramp-{repeat}.txt')['BenchmarkMNIST'] for repeat in range(1, 8)]
        ramp[variant] = dict(latency_us=stats([row['latency_us']['median'] for row in rows]),
            bytes=[row['bytes'][0] for row in rows], allocs=[row['allocs'][0] for row in rows])
    diagnostics = {path.stem: benchmarks(path) for path in sorted(OUT.glob('*.txt'))
        if path.name.endswith(('-amortization.txt', '-cold-session.txt', '-workloads.txt', '-concurrent-2.txt',
            '-pipeline.txt', '-nodes.txt', '-lifecycles.txt', '-relu.txt', '-publication.txt', '-load.txt'))}
    metal = {path.stem: dict(skips=len(re.findall(r'--- SKIP:', path.read_text())),
        passes=len(re.findall(r'--- PASS:', path.read_text()))) for path in sorted(OUT.glob('*metal*.txt'))}
    commands = [json.loads(line) for line in (OUT / 'commands.jsonl').read_text().splitlines()]
    summary = dict(environment=json.loads((OUT / 'environment.json').read_text()),
        comparison_environment=reports['candidate'][0]['environment'],
        fresh_baseline=json.loads((OUT / 'fresh-baseline.json').read_text())['timing'],
        prototype_v2_ramp={variant: stats([benchmarks(path)['BenchmarkMNIST']['latency_us']['median']
            for path in sorted(OUT.glob('v2-prototype-' + variant + '-ramp-*.txt'))])
            for variant in ['control', 'prototype']},
        production_commit='7525766659926bc9e47f4100de3821c676e4ba12',
        fixed_policy=fixed, go_difference_us=go_difference,
        go_improvement_pct=100*go_difference/fixed['control']['go']['median'],
        ort_movement_us=ort_movement,
        ort_movement_pct=100*ort_movement/fixed['control']['ort']['median'],
        paired_go_differences_us=stats(paired), ramp=ramp, diagnostics=diagnostics,
        retained=dict(logical_payload=83664, tensor_allocator_volume=91449, tensors=9,
            tensor_objects=36, pointer_table_bytes=96, pointer_table_objects=1,
            logical_capacity_bound=121296, tensor_capacity_bound=11,
            minimum_live=335855448, session_allocator_bytes=80, previous_session_allocator_bytes=24),
        roadmap={str(ratio): dict(go_target_us=ratio*fixed['candidate']['ort']['median'],
            further_reduction_us=max(0, fixed['candidate']['go']['median']-ratio*fixed['candidate']['ort']['median']))
            for ratio in [1.25, 1.1]}, metal=metal)
    paths = [*sorted(ROOT.glob('*.go')), ROOT / 'go.mod', ROOT / 'go.sum', ROOT / 'tools/compare_runtime.py',
        *sorted((ROOT / 'tools/compare_runtime').glob('*.go'))]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (DATA / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (DATA / 'validation.json').write_text(json.dumps(commands, indent=2) + '\n')
    evidence = dict(summary=summary, commands=commands, reports=reports,
        other_reports={path.name: json.loads(path.read_text()) for path in OUT.glob('*.json')
            if path.name in ['fresh-baseline.json', 'purego-comparison.json']},
        logs={path.name: path.read_text() for path in sorted(OUT.glob('*.txt'))},
        final_source_sha256=hashes,
        measured_source_sha256=json.loads((OUT / 'candidate-sha256.json').read_text()),
        diagnostic_sources={path.name: path.read_text() for path in sorted(HERE.iterdir())
            if path.suffix in ['.py', '.txt']},
        original_source_sha256=json.loads((OUT / 'source-sha256.json').read_text()),
        snapshots={str(path.relative_to(OUT)): path.read_text()
            for directory in ['original', 'prototype-v1', 'prototype', 'measured-source']
            for path in sorted((OUT / directory).glob('*.go'))})
    # Hardware serial/UUID values never enter the published evidence.
    evidence['logs'].pop('hardware.txt', None)
    payload = json.dumps(evidence, separators=(',', ':')).encode()
    (DATA / 'evidence.json.gz').write_bytes(gzip.compress(payload, mtime=0))
    print(json.dumps({key: summary[key] for key in ['fixed_policy', 'go_difference_us',
        'go_improvement_pct', 'ort_movement_us', 'paired_go_differences_us', 'ramp', 'roadmap', 'metal']}, indent=2))
