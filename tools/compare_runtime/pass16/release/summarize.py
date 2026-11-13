"""Package the published Pass 16 integration independently of its investigation."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / '.cache/compare-runtime/pass16-release'
DATA = HERE / 'data'


def stats(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values),
        cv_pct=100 * statistics.pstdev(values) / statistics.mean(values) if statistics.mean(values) > 0 else None,
        values=values)


def benchmarks(path):
    rows = {}
    for match in re.finditer(
            r'^(Benchmark\S+)\s+\d+\s+([\d.]+) ns/op\s+(\d+) B/op\s+(\d+) allocs/op',
            path.read_text(), re.M):
        name, latency, size, count = match.groups()
        rows.setdefault(name, []).append((float(latency) / 1000, int(size), int(count)))
    assert rows, path
    return {name: dict(stats([row[0] for row in samples]),
                      bytes=sorted(set(row[1] for row in samples)),
                      allocs=sorted(set(row[2] for row in samples))) for name, samples in rows.items()}


def validate_report(report, expected, outputs):
    assert report['passed'] and len(report['inputs']) == 19
    assert report['timing']['warmup_per_case'] == 10
    assert report['timing']['iterations_per_case'] == 100
    assert report['timing']['go']['samples'] == report['timing']['ort']['samples'] == 1900
    assert report['comparison']['atol'] == 1e-4 and report['comparison']['rtol'] == 1e-5
    assert dict(inputs=report['inputs'], model=report['model']['sha256'],
                driver=report['environment']['driver_sha256'],
                runner=report['environment']['runner_sha256']) == expected
    current = {(row['case'], row['output']): row['go_output'] for row in report['results']}
    if outputs is not None:
        assert current == outputs
    return current


def main():
    summary = {}
    evidence = dict(reports={}, logs={})
    expected = json.loads((ROOT / 'tools/compare_runtime/pass15/data/summary.json').read_text())['fingerprints']
    outputs = None
    for variant, version in [('control', 'v0.10.9'), ('candidate', 'v0.10.10')]:
        paths = sorted(OUT.glob(variant + '-comparison-*.json'))
        assert len(paths) == 7
        reports = [json.loads(path.read_text()) for path in paths]
        for path, report in zip(paths, reports):
            outputs = validate_report(report, expected, outputs)
            evidence['reports'][path.name] = report
        timing = {engine: stats([report['timing'][engine]['median_us'] for report in reports])
                  for engine in ['go', 'ort']}
        ramps = [next(iter(benchmarks(path).values())) for path in sorted(OUT.glob(variant + '-ramp-*.txt'))]
        assert len(ramps) == 7
        assert all(row['bytes'] == ([92874] if variant == 'control' else [92826]) for row in ramps)
        assert all(row['allocs'] == ([73] if variant == 'control' else [72]) for row in ramps)
        module = json.loads((OUT / ('module-' + variant + '.txt')).read_text())
        assert module['Version'] == version and 'Replace' not in module
        nodes = benchmarks(OUT / (variant + '-nodes.txt'))
        assert all(len(row['values']) == 7 for row in nodes.values())
        assert nodes['BenchmarkMNISTNodes/11_Add']['bytes'] == ([336] if variant == 'control' else [288])
        assert nodes['BenchmarkMNISTNodes/11_Add']['allocs'] == ([7] if variant == 'control' else [6])
        assert nodes['BenchmarkMNISTNodes/10_MatMul']['bytes'] == [144]
        assert nodes['BenchmarkMNISTNodes/10_MatMul']['allocs'] == [3]
        summary[variant] = dict(timing=timing, ratio=timing['go']['median'] / timing['ort']['median'],
            ramp=dict(stats([row['median'] for row in ramps]), bytes=ramps[0]['bytes'][0], allocs=ramps[0]['allocs'][0]),
            nodes=nodes, load=benchmarks(OUT / (variant + '-load.txt')), module=module)
    purego_path = OUT / 'candidate-purego.json'
    purego = json.loads(purego_path.read_text())
    validate_report(purego, expected, outputs)
    evidence['reports'][purego_path.name] = purego
    summary['saving_us'] = summary['control']['timing']['go']['median'] - summary['candidate']['timing']['go']['median']
    summary['saving_pct'] = 100 * summary['saving_us'] / summary['control']['timing']['go']['median']
    summary['paired_savings_us'] = stats([a - b for a, b in zip(
        summary['control']['timing']['go']['values'], summary['candidate']['timing']['go']['values'])])
    summary['paired_savings_us']['cv_pct'] = None
    summary['maximum_absolute_error'] = max(row['go_vs_ort']['max_absolute_error']
        for report in evidence['reports'].values() for row in report['results'])
    summary['go_outputs_bit_identical'] = True
    summary['purego_comparison_passed'] = purego['passed']
    summary['environment'] = json.loads((OUT / 'environment.json').read_text())
    summary['fingerprints'] = expected
    hashes = json.loads((OUT / 'source-sha256.json').read_text())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    summary['source_sha256'] = hashes
    records = [json.loads(line) for line in (OUT / 'commands.jsonl').read_text().splitlines()]
    failures = [record for record in records if record['exit_code'] != 0]
    # Preserve the observed native Metal stress failure and require the complete
    # predefined follow-up checks; never discard it or count it as a passing run.
    for failure in failures:
        assert failure['name'] == 'backend-test-metal-host'
        failed_log = (OUT / (failure['name'] + '.txt')).read_text()
        assert failed_log.count('--- FAIL:') == 1
        assert '--- FAIL: Test_SequentialResidentLongRunningMixedStress' in failed_log
    summary['validation_failures'] = failures
    latest = {record['name']: record for record in records}
    if failures:
        for name in ['control-metal-stress-repeat', 'candidate-metal-stress-repeat',
                     'control-metal-serial-suite', 'candidate-metal-serial-suite']:
            assert latest[name]['exit_code'] == 0
    for name in ['default', 'no-cgo', 'purego', 'race', 'purego-race', 'metal']:
        assert latest['test-' + name]['exit_code'] == 0
        assert latest['backend-test-' + name]['exit_code'] == 0
    detail = OUT / 'backend-metal-host-detail.txt'
    if not detail.exists():
        detail = OUT / 'backend-metal-detail.txt'
    summary['metal_validation'] = dict(detail=detail.name,
        skipped_tests=len(re.findall(r'^--- SKIP:', detail.read_text(), re.M)))
    summary['replay_source_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [HERE / 'measure.py', HERE / 'summarize.py', ROOT / 'tools/compare_runtime/pass15/measure.py']}
    for path in OUT.glob('*.txt'):
        evidence['logs'][path.name] = path.read_text()
    DATA.mkdir(exist_ok=True)
    (DATA / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (DATA / 'validation.json').write_text(json.dumps(records, indent=2) + '\n')
    (DATA / 'evidence.json.gz').write_bytes(gzip.compress(
        json.dumps(evidence, separators=(',', ':')).encode(), mtime=0))
    print(json.dumps({key: summary[key] for key in ['saving_us', 'saving_pct', 'paired_savings_us']}, indent=2))
    for variant in ['control', 'candidate']:
        print(variant, json.dumps({key: summary[variant][key] for key in ['timing', 'ratio', 'ramp', 'nodes', 'load']}, indent=2))


if __name__ == '__main__':
    main()
