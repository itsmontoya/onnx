"""Preserve released-module results without pooling earlier local experiments."""
import gzip
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT/'.cache/compare-runtime/pass15-release'
DATA = HERE/'data'
DATA.mkdir(exist_ok=True)


def stats(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values),
        cv_pct=100*statistics.stdev(values)/statistics.mean(values) if len(values)>1 and statistics.mean(values)>0 else 0,
        values=values)


def benchmarks(path):
    rows = {}
    for line in path.read_text().splitlines():
        match = re.match(r'(Benchmark\S+)\s+\d+\s+([\d.]+) ns/op(?:\s+[\d.]+ MB/s)?\s+(\d+) B/op\s+(\d+) allocs/op',line)
        if match:
            name, ns, size, count = match.groups()
            rows.setdefault(name,[]).append((float(ns),int(size),int(count)))
    assert rows,path
    return {name:dict(stats([row[0] for row in samples]),
                     bytes=statistics.median([row[1] for row in samples]),
                     allocs=statistics.median([row[2] for row in samples])) for name,samples in rows.items()}


summary = {}
evidence = dict(reports={},logs={})
expected = json.loads((HERE.parent/'data/summary.json').read_text())['fingerprints']
outputs = None
for variant in ['control','candidate']:
    paths = sorted(OUT.glob(variant+'-comparison-*.json'))
    assert len(paths)==7
    reports = [json.loads(path.read_text()) for path in paths]
    for path,report in zip(paths,reports):
        assert report['passed'] and len(report['inputs'])==19
        assert report['timing']['warmup_per_case']==10 and report['timing']['iterations_per_case']==100
        assert report['timing']['go']['samples']==report['timing']['ort']['samples']==1900
        assert report['comparison']['atol']==1e-4 and report['comparison']['rtol']==1e-5
        assert dict(inputs=report['inputs'],model=report['model']['sha256'],
            driver=report['environment']['driver_sha256'],runner=report['environment']['runner_sha256'])==expected
        current = {(row['case'],row['output']):row['go_output'] for row in report['results']}
        if outputs is None:
            outputs = current
        assert outputs==current
        evidence['reports'][path.name]=report
    timing = {engine:stats([report['timing'][engine]['median_us'] for report in reports]) for engine in ['go','ort']}
    ramps = [next(iter(benchmarks(path).values())) for path in sorted(OUT.glob(variant+'-ramp-*.txt'))]
    nodes = benchmarks(OUT/(variant+'-nodes.txt'))
    groups = {}
    for name,row in nodes.items():
        parts = name.split('/')
        if len(parts)==3 and parts[-1]=='WithCorrection':
            operator='Relu'
        elif len(parts)==2 and parts[1].split('_',1)[1] in ['Conv','Add','MaxPool','MatMul']:
            operator=parts[1].split('_',1)[1]
        else:
            continue
        groups[operator]=groups.get(operator,0)+row['median']/1000
    module = json.loads((OUT/('module-'+variant+'.txt')).read_text())
    assert 'Replace' not in module
    summary[variant]=dict(timing=timing,ratio=timing['go']['median']/timing['ort']['median'],
        ramp=dict(stats([row['median']/1000 for row in ramps]),bytes=ramps[0]['bytes'],allocs=ramps[0]['allocs']),
        nodes=nodes,node_sums_us=groups,load=benchmarks(OUT/(variant+'-load.txt')),module=module)
summary['saving_us']=summary['control']['timing']['go']['median']-summary['candidate']['timing']['go']['median']
summary['saving_pct']=100*summary['saving_us']/summary['control']['timing']['go']['median']
summary['paired_savings_us']=stats([a-b for a,b in zip(summary['control']['timing']['go']['values'],summary['candidate']['timing']['go']['values'])])
summary['paired_savings_us']['cv_pct']=None
reference=summary['candidate']['timing']['ort']['median']
summary['roadmap']={str(factor):dict(target_us=factor*reference,
    remaining_us=summary['candidate']['timing']['go']['median']-factor*reference) for factor in [1.25,1.10]}
summary['max_absolute_error']=max(row['go_vs_ort']['max_absolute_error'] for report in evidence['reports'].values() for row in report['results'])
summary['go_outputs_bit_identical']=True
summary['integration_commit']='3b37416360c1ef9ef98c8bf38f011897d334bb58'
for name in ['environment','release-provenance','source-identity']:
    summary[name]=json.loads((OUT/(name+'.json')).read_text())
ort=json.loads((OUT/'ort-node-samples.json').read_text())
groups={}
for nodes in ort['repetitions']:
    totals={}
    for name,values in nodes.items():
        operator=('Conv' if name.startswith('Convolution') else 'Relu' if name.startswith('ReLU')
                  else 'MaxPool' if name.startswith('Pooling') else 'Add' if name.startswith('Plus')
                  else 'Reshape' if 'reshape' in name else 'MatMul')
        totals[operator]=totals.get(operator,0)+statistics.mean(values)
    for operator,value in totals.items():
        groups.setdefault(operator,[]).append(value)
summary['ort_node_sums_us']={name:stats(values) for name,values in groups.items()}
evidence['ort_node_samples']=ort
for path in OUT.glob('*.txt'):
    if path.name == 'summary-print.txt':
        continue
    evidence['logs'][path.name]=path.read_text()
records=[json.loads(line) for line in (OUT/'commands.jsonl').read_text().splitlines()]
assert all(record['exit_code']==0 for record in records)
(DATA/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(DATA/'validation.json').write_text(json.dumps(records,indent=2)+'\n')
(DATA/'evidence.json.gz').write_bytes(gzip.compress(json.dumps(evidence,separators=(',',':')).encode(),mtime=0))
print(json.dumps({key:summary[key] for key in ['saving_us','saving_pct','paired_savings_us','roadmap']},indent=2))
for variant in ['control','candidate']:
    print(variant,json.dumps({key:summary[variant][key] for key in ['timing','ratio','ramp','node_sums_us','load']},indent=2))
