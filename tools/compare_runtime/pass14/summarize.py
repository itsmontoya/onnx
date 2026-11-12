"""Publish the fixed estimators and compact, lossless measurement evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'.cache/compare-runtime/pass14-release'
DATA = HERE/'data'
DATA.mkdir(exist_ok=True)


def stats(values):
    return dict(median=statistics.median(values),minimum=min(values),maximum=max(values),
                cv_pct=100*statistics.stdev(values)/statistics.mean(values) if len(values)>1 else 0,
                samples=len(values),values=values)


def benchmarks(path):
    rows = {}
    for line in path.read_text().splitlines():
        match = re.match(r'(Benchmark\S+)\s+\d+\s+([\d.]+) ns/op(?:\s+[\d.]+ MB/s)?\s+(\d+) B/op\s+(\d+) allocs/op',line)
        if match:
            name, ns, size, count = match.groups()
            rows.setdefault(name,[]).append(dict(ns=float(ns),bytes=int(size),allocs=int(count)))
    assert rows, path
    return {name:dict(stats([row['ns'] for row in samples]),bytes=samples[0]['bytes'],allocs=samples[0]['allocs'])
            for name,samples in rows.items()}


summary = {}
evidence = {'reports':{},'benchmarks':{},'validation_logs':{}}
fingerprints = None
expected_go_outputs = None
for variant in ['control','candidate']:
    paths = sorted(OUT.glob(variant+'-comparison-*.json'))
    assert len(paths)==7
    reports = [json.loads(path.read_text()) for path in paths]
    evidence['reports'].update({path.name:report for path,report in zip(paths,reports)})
    for report in reports:
        assert report['passed'] and len(report['inputs'])==19
        assert report['timing']['warmup_per_case']==10 and report['timing']['iterations_per_case']==100
        assert report['timing']['go']['samples']==report['timing']['ort']['samples']==1900
        assert report['comparison']['atol']==1e-4 and report['comparison']['rtol']==1e-5
        current = dict(inputs=report['inputs'],model=report['model']['sha256'],
                       driver=report['environment']['driver_sha256'],runner=report['environment']['runner_sha256'])
        if fingerprints is None:
            fingerprints = current
        assert fingerprints==current
        outputs = {(row['case'],row['output']):row['go_output'] for row in report['results']}
        if expected_go_outputs is None:
            expected_go_outputs = outputs
        assert outputs==expected_go_outputs
    timing = {engine:dict(stats([report['timing'][engine]['median_us'] for report in reports]),
                within_report_p95_median_us=statistics.median([report['timing'][engine]['p95_us'] for report in reports]))
              for engine in ['go','ort']}
    ramp_paths = sorted(OUT.glob(variant+'-ramp-*.txt'))
    assert len(ramp_paths)==7
    ramps = [next(iter(benchmarks(path).values())) for path in ramp_paths]
    nodes = benchmarks(OUT/(variant+'-nodes.txt'))
    groups = {}
    backend_relu = []
    for name,row in nodes.items():
        parts = name.split('/')
        if len(parts)==3 and parts[-1]=='PreservingBackendOnly':
            backend_relu.append(row['median']/1000)
        if len(parts)==3 and parts[-1]=='WithCorrection':
            operator = 'Relu'
        elif len(parts)==2 and parts[1].split('_',1)[1] in ['Conv','Add','MaxPool','MatMul']:
            operator = parts[1].split('_',1)[1]
        else:
            continue
        groups[operator] = groups.get(operator,0)+row['median']/1000
    summary[variant] = dict(timing=timing,ratio=timing['go']['median']/timing['ort']['median'],
        ramp=dict(stats([row['median']/1000 for row in ramps]),bytes=ramps[0]['bytes'],allocs=ramps[0]['allocs']),
        nodes=nodes,node_sums_us=groups,preserving_backend_relu_sum_us=sum(backend_relu),
        module=json.loads((OUT/('module-'+variant+'.txt')).read_text()),environment=reports[0]['environment'])
    for path in [*ramp_paths,OUT/(variant+'-nodes.txt')]:
        evidence['benchmarks'][path.name] = path.read_text()

summary['saving_us'] = summary['control']['timing']['go']['median']-summary['candidate']['timing']['go']['median']
summary['saving_pct'] = 100*summary['saving_us']/summary['control']['timing']['go']['median']
summary['paired_savings_us'] = stats([a-b for a,b in zip(summary['control']['timing']['go']['values'],summary['candidate']['timing']['go']['values'])])
summary['remaining_to_125_us'] = {variant:summary[variant]['timing']['go']['median']-1.25*summary[variant]['timing']['ort']['median'] for variant in ['control','candidate']}
summary['max_absolute_error'] = max(row['go_vs_ort']['max_absolute_error'] for report in evidence['reports'].values() for row in report['results'])
summary['go_outputs_bit_identical'] = True
summary['source_sha256'] = json.loads((OUT/'source-sha256.json').read_text())
summary['release_provenance'] = json.loads((OUT/'release-provenance.txt').read_text())
summary['host_environment'] = json.loads((OUT/'environment.json').read_text())
release_dir = Path(summary['release_provenance']['Dir'])
summary['released_relu_sha256'] = {str(path.relative_to(release_dir)):hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [release_dir/'tensor/relu_preserving_nan.go',release_dir/'tensor/relu_preserving_nan_arm64.go',
                 release_dir/'tensor/relu_preserving_nan_arm64.s',release_dir/'tensor/relu_preserving_nan_default.go']}
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
records = [json.loads(line) for line in (OUT/'commands.jsonl').read_text().splitlines()]
assert all(record['exit_code']==0 for record in records)
for record in records:
    if record['name'].startswith(('test-','fuzz-','build-','vet','tidy','verify','format','diff-check')):
        evidence['validation_logs'][record['name']] = (OUT/(record['name']+'.txt')).read_text()
for name in ['candidate-cpu-top','candidate-alloc_space','candidate-alloc_objects',
             'control-relu-objdump','candidate-relu-objdump']:
    evidence['validation_logs'][name] = (OUT/(name+'.txt')).read_text()
(DATA/'validation.json').write_text(json.dumps(records,indent=2)+'\n')
(DATA/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(DATA/'evidence.json.gz').write_bytes(gzip.compress(json.dumps(evidence,separators=(',',':')).encode(),mtime=0))
print(json.dumps({key:summary[key] for key in ['saving_us','saving_pct','paired_savings_us','remaining_to_125_us','max_absolute_error']},indent=2))
for variant in ['control','candidate']:
    print(variant,json.dumps({key:summary[variant][key] for key in ['timing','ratio','ramp','node_sums_us','preserving_backend_relu_sum_us']},indent=2))
