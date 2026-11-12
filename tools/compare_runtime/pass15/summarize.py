"""Keep fixed estimators and all valid Pass 15 comparison samples."""
import gzip
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'.cache/compare-runtime/pass15'
DATA = Path(__file__).resolve().parent/'data'
DATA.mkdir(exist_ok=True)


def stats(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values),
                cv_pct=100*statistics.stdev(values)/statistics.mean(values) if len(values)>1 and statistics.mean(values) != 0 else 0,
                samples=len(values), values=values)


def benchmarks(path):
    rows = {}
    for line in path.read_text().splitlines():
        match = re.match(r'(Benchmark\S+)\s+\d+\s+([\d.]+) ns/op(?:\s+[\d.]+ MB/s)?\s+(\d+) B/op\s+(\d+) allocs/op', line)
        if match:
            name, ns, size, count = match.groups()
            rows.setdefault(name, []).append(dict(ns=float(ns), bytes=int(size), allocs=int(count)))
    assert rows, path
    return {name:dict(stats([row['ns'] for row in samples]),
                     bytes=stats([row['bytes'] for row in samples]),
                     allocs=stats([row['allocs'] for row in samples])) for name,samples in rows.items()}


evidence = dict(reports={}, logs={}, source={})
summary = {}
fingerprints = None
expected_outputs = None
for stage, directory in [('initial', OUT/'initial'), ('final', OUT)]:
    result = {}
    for variant in ['control','candidate']:
        paths = sorted(directory.glob(variant+'-comparison-*.json'), key=lambda path:int(path.stem.rsplit('-',1)[1]))
        assert len(paths) == (7 if stage == 'initial' else 14), (stage, variant, len(paths))
        reports = [json.loads(path.read_text()) for path in paths]
        for path, report in zip(paths, reports):
            assert report['passed'] and len(report['inputs']) == 19
            assert report['timing']['warmup_per_case'] == 10 and report['timing']['iterations_per_case'] == 100
            assert report['timing']['go']['samples'] == report['timing']['ort']['samples'] == 1900
            assert report['comparison']['atol'] == 1e-4 and report['comparison']['rtol'] == 1e-5
            current = dict(inputs=report['inputs'], model=report['model']['sha256'],
                           driver=report['environment']['driver_sha256'], runner=report['environment']['runner_sha256'])
            if fingerprints is None:
                fingerprints = current
            assert fingerprints == current
            outputs = {(row['case'],row['output']):row['go_output'] for row in report['results']}
            if expected_outputs is None:
                expected_outputs = outputs
            assert expected_outputs == outputs
            evidence['reports'][stage+'/'+path.name] = report
        timing = {engine:stats([report['timing'][engine]['median_us'] for report in reports]) for engine in ['go','ort']}
        ramp = [next(iter(benchmarks(path).values())) for path in sorted(directory.glob(variant+'-ramp-*.txt'))]
        nodes = benchmarks(directory/(variant+'-nodes.txt'))
        groups = {}
        for name,row in nodes.items():
            parts = name.split('/')
            if len(parts) == 3 and parts[-1] == 'WithCorrection':
                operator = 'Relu'
            elif len(parts) == 2 and parts[1].split('_',1)[1] in ['Conv','Add','MaxPool','MatMul']:
                operator = parts[1].split('_',1)[1]
            else:
                continue
            groups[operator] = groups.get(operator,0)+row['median']/1000
        result[variant] = dict(timing=timing, ratio=timing['go']['median']/timing['ort']['median'],
            ramp=dict(stats([row['median']/1000 for row in ramp]), bytes=ramp[0]['bytes'], allocs=ramp[0]['allocs']),
            nodes=nodes, node_sums_us=groups, load=benchmarks(directory/(variant+'-load.txt')))
    control, candidate = result['control'], result['candidate']
    result['saving_us'] = control['timing']['go']['median']-candidate['timing']['go']['median']
    result['saving_pct'] = 100*result['saving_us']/control['timing']['go']['median']
    result['paired_savings_us'] = stats([a-b for a,b in zip(control['timing']['go']['values'],candidate['timing']['go']['values'])])
    result['paired_savings_us']['cv_pct'] = None  # Differences straddle zero; CV is not useful.
    result['roadmap'] = {str(factor):dict(target_us=factor*candidate['timing']['ort']['median'],
        remaining_us=candidate['timing']['go']['median']-factor*candidate['timing']['ort']['median']) for factor in [1.25,1.10]}
    summary[stage] = result

summary['prototype'] = {}
for variant in ['ownership','prepared']:
    paths = sorted(OUT.glob('prototype-'+variant+'-*.json'))
    assert len(paths) == 7
    reports = [json.loads(path.read_text()) for path in paths]
    for path,report in zip(paths,reports):
        assert report['passed'] and len(report['inputs']) == 19
        assert report['timing']['go']['samples'] == report['timing']['ort']['samples'] == 1900
        assert report['timing']['warmup_per_case'] == 10 and report['timing']['iterations_per_case'] == 100
        assert report['comparison']['atol'] == 1e-4 and report['comparison']['rtol'] == 1e-5
        assert dict(inputs=report['inputs'],model=report['model']['sha256'],driver=report['environment']['driver_sha256'],runner=report['environment']['runner_sha256']) == fingerprints
        assert {(row['case'],row['output']):row['go_output'] for row in report['results']} == expected_outputs
        evidence['reports'][path.name] = report
    summary['prototype'][variant] = {engine:stats([report['timing'][engine]['median_us'] for report in reports]) for engine in ['go','ort']}
summary['prototype']['components'] = benchmarks(OUT/'prepared-components.txt')
summary['prototype']['ramp'] = benchmarks(OUT/'prepared-ramp.txt')
preflight = json.loads((OUT/'baseline-initial.json').read_text())
assert preflight['passed']
evidence['preflight_report'] = preflight
summary['max_absolute_error'] = max(row['go_vs_ort']['max_absolute_error'] for report in evidence['reports'].values() for row in report['results'])
summary['go_outputs_bit_identical'] = True
summary['fingerprints'] = fingerprints
summary['backend_candidate_commit'] = (OUT/'backend-candidate-commit.txt').read_text().strip()
summary['environment'] = json.loads((OUT/'environment.json').read_text())
summary['source_sha256'] = json.loads((OUT/'source-sha256.json').read_text())
summary['diagnostics'] = benchmarks(OUT/'diagnostics.txt')
summary['backend_dynamic'] = {variant:benchmarks(OUT/(variant+'-dynamic.txt')) for variant in ['control','candidate']}
summary['onnx_dynamic'] = {variant:benchmarks(OUT/(variant+'-dynamic-onnx.txt')) for variant in ['control','candidate']}
summary['machinery'] = benchmarks(OUT/'candidate-machinery.txt')
ort = json.loads((OUT/'ort-node-samples.json').read_text())
groups = {}
for nodes in ort['repetitions']:
    totals = {}
    for name,values in nodes.items():
        operator = ('Conv' if name.startswith('Convolution') else 'Relu' if name.startswith('ReLU')
                    else 'MaxPool' if name.startswith('Pooling') else 'Add' if name.startswith('Plus')
                    else 'Reshape' if 'reshape' in name else 'MatMul')
        totals[operator] = totals.get(operator,0)+statistics.mean(values)
    for operator,value in totals.items():
        groups.setdefault(operator,[]).append(value)
summary['ort_node_sums_us'] = {operator:stats(values) for operator,values in groups.items()}
evidence['ort_node_samples'] = ort
for directory in [OUT, OUT/'initial']:
    for path in directory.glob('*.txt'):
        if path.name == 'summary-print.txt':
            continue
        evidence['logs'][str(path.relative_to(OUT))] = path.read_text()
for path in Path(__file__).resolve().parent.glob('*.txt'):
    evidence['source'][path.name] = path.read_text()
records = [json.loads(line) for line in (OUT/'commands.jsonl').read_text().splitlines()]
last = {record['name']:record for record in records}
assert all(record['exit_code'] == 0 for record in last.values())
(DATA/'validation.json').write_text(json.dumps(records, indent=2)+'\n')
(DATA/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
(DATA/'evidence.json.gz').write_bytes(gzip.compress(json.dumps(evidence, separators=(',',':')).encode(), mtime=0))
print(json.dumps({key:summary['final'][key] for key in ['saving_us','saving_pct','paired_savings_us','roadmap']}, indent=2))
for variant in ['control','candidate']:
    print(variant, json.dumps({key:summary['final'][variant][key] for key in ['timing','ratio','ramp','node_sums_us','load']}, indent=2))
