"""Archive every Pass 20 observation, including diagnostic failures and outliers."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics as s

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass20', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
OUT, ROOT, DATA = p.m.OUT, p.ROOT, HERE / 'data'


def stats(values):
    return dict(median=s.median(values), minimum=min(values), maximum=max(values),
        cv_pct=100 * s.pstdev(values) / s.mean(values), values=values)


def benchmarks(path):
    rows = {}
    for name, elapsed, allocated, count in re.findall(
            r'^(Benchmark\S+)\s+\d+\s+([\d.]+) ns/op\s+(\d+) B/op\s+(\d+) allocs/op', path.read_text(), re.M):
        row = rows.setdefault(name, dict(latency_us=[], bytes=[], allocs=[]))
        row['latency_us'].append(float(elapsed) / 1000)
        row['bytes'].append(int(allocated))
        row['allocs'].append(int(count))
    return {name: dict(latency_us=stats(row['latency_us']), bytes=row['bytes'], allocs=row['allocs'])
        for name, row in rows.items()}


def comparisons(prefix, variants):
    reports = {variant: [json.loads((OUT / f'{prefix}{variant}-comparison-{repeat}.json').read_text())
        for repeat in range(1, 8)] for variant in variants}
    summary = {variant: {engine: stats([report['timing'][engine]['median_us'] for report in samples])
        for engine in ['go', 'ort']} for variant, samples in reports.items()}
    for row in summary.values():
        row['ratio'] = row['go']['median'] / row['ort']['median']
    if len(variants) == 2:
        control, candidate = variants
        difference = summary[control]['go']['median'] - summary[candidate]['go']['median']
        movement = summary[candidate]['ort']['median'] - summary[control]['ort']['median']
        summary['movement'] = dict(go_reduction_us=difference,
            go_reduction_pct=100 * difference / summary[control]['go']['median'],
            ort_increase_us=movement, ort_increase_pct=100 * movement / summary[control]['ort']['median'],
            paired_go_reduction_us=stats([left['timing']['go']['median_us'] - right['timing']['go']['median_us']
                for left, right in zip(reports[control], reports[candidate])]))
    return summary, reports


if __name__ == '__main__':
    DATA.mkdir(exist_ok=True)
    baseline, baseline_reports = comparisons('', ['baseline'])
    production, production_reports = comparisons('production-', ['merged', 'candidate'])
    diagnostics = {path.stem: benchmarks(path) for path in sorted(OUT.glob('*.txt')) if benchmarks(path)}
    ablations = {}
    for variant in ['control', 'local-dtype', 'prepared-proof', 'invalid-direct', 'shell']:
        rows = [diagnostics[f'{variant}-ablation-{repeat}']['BenchmarkPass19Run'] for repeat in range(1, 8)]
        ablations[variant] = dict(latency_us=stats([row['latency_us']['median'] for row in rows]),
            bytes=[row['bytes'][0] for row in rows], allocs=[row['allocs'][0] for row in rows])
    ramps = {}
    for prefix, variants in [('', ['baseline']), ('production-', ['merged', 'candidate'])]:
        for variant in variants:
            rows = [diagnostics[f'{prefix}{variant}-ramp-{repeat}']['BenchmarkMNIST'] for repeat in range(1, 8)]
            ramps[prefix + variant] = dict(latency_us=stats([row['latency_us']['median'] for row in rows]),
                bytes=[row['bytes'][0] for row in rows], allocs=[row['allocs'][0] for row in rows])
    frequency = {}
    for variant in ['counter', 'counter-candidate']:
        text = (OUT / ('accessor-frequency-' + variant + '.txt')).read_text()
        frequency[variant] = {}
        for label in ['COUNTS', 'SITES', 'MATRIX']:
            observations = [json.loads(line) for line in re.findall(label + r' (\{[^\n]*\})', text)]
            assert len(observations) == 2 and observations[0] == observations[1]
            frequency[variant][label.lower()] = observations[0]
    metal = {path.stem: dict(skips=len(re.findall(r'--- SKIP:', path.read_text())),
        passes=len(re.findall(r'--- PASS:', path.read_text())), failures=len(re.findall(r'--- FAIL:', path.read_text())))
        for path in OUT.glob('*metal*.txt')}
    summary = dict(environment=json.loads((OUT / 'environment.json').read_text()),
        implementation_commit=(OUT / 'production-commit.txt').read_text().strip()
            if (OUT / 'production-commit.txt').exists() else None,
        fixed_policy_environment=baseline_reports['baseline'][0]['environment'],
        model=baseline_reports['baseline'][0]['model'], baseline=baseline, production=production,
        ramps=ramps, ablations=ablations, diagnostics=diagnostics, frequency=frequency, metal=metal,
        decision='One ONNX-only optimization: reuse actual dtype within each value check; no backend changes.',
        roadmap={str(ratio): dict(go_target_us=ratio * production['candidate']['ort']['median'],
            further_reduction_us=max(0, production['candidate']['go']['median'] - ratio * production['candidate']['ort']['median']))
            for ratio in [1.25, 1.1]})
    commands = [json.loads(line) for line in (OUT / 'commands.jsonl').read_text().splitlines()]
    (DATA / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (DATA / 'validation.json').write_text(json.dumps(commands, indent=2) + '\n')
    backend, reference = p.m.BACKEND, OUT / 'backend-reference'
    for original in backend.rglob('*'):
        if original.is_file():
            assert original.read_bytes() == (reference / original.relative_to(backend)).read_bytes()
    for name, digest in json.loads((OUT / 'backend-sha256.json').read_text()).items():
        assert hashlib.sha256((backend / name).read_bytes()).hexdigest() == digest, name
    sources = [*sorted(ROOT.glob('*.go')), ROOT / 'go.mod', ROOT / 'go.sum', ROOT / 'tools/compare_runtime.py',
        *sorted((ROOT / 'tools/compare_runtime').glob('*.go'))]
    final_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for name, digest in json.loads((OUT / 'candidate-sha256.json').read_text()).items():
        assert final_hashes[name] == digest, name
    evidence = dict(summary=summary, commands=commands,
        reports=dict(baseline=baseline_reports, production=production_reports),
        purego=json.loads((OUT / 'purego-comparison.json').read_text()),
        logs={path.name: path.read_text() for path in sorted(OUT.glob('*.txt')) if path.name != 'hardware.txt'},
        original_source_sha256=json.loads((OUT / 'source-sha256.json').read_text()),
        measured_candidate_sha256=json.loads((OUT / 'candidate-sha256.json').read_text()),
        final_source_sha256=final_hashes,
        backend_sha256=json.loads((OUT / 'backend-sha256.json').read_text()),
        diagnostic_sources={path.name: path.read_text() for path in sorted(HERE.iterdir())
            if path.suffix in ['.py', '.txt']},
        overlays={path.name: json.loads(path.read_text()) for path in OUT.glob('*overlay.json')},
        source_snapshots={path.name: path.read_text() for path in OUT.glob('*.go')})
    (DATA / 'evidence.json.gz').write_bytes(gzip.compress(json.dumps(evidence, separators=(',', ':')).encode(), mtime=0))
    for path in OUT.glob('*.pprof'):
        (DATA / path.name).write_bytes(path.read_bytes())
    print(json.dumps({key: summary[key] for key in ['baseline', 'production', 'ramps', 'roadmap', 'metal']}, indent=2))
