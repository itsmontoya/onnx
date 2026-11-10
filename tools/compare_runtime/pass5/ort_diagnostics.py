#!/usr/bin/env python3
"""Separate ORT experiments; never changes the comparison driver's settings.

Run using the pinned comparison Python environment, with an output directory
as the sole argument. Timings use the deterministic Go benchmark input, not
the comparison driver's 19-case sample distribution.
"""
import sys
import collections, json, statistics, time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
path=ROOT/'.cache/compare-runtime/mnist-ir8-opset13.onnx'
model=onnx.load(path)
x=(np.arange(784,dtype=np.int64)%256).astype(np.float32).reshape(1,1,28,28)/np.float32(255)
feeds={'Input3':x}
def session(name, enabled=False, arena=True, pattern=True, threads=1, profile=False):
    o=ort.SessionOptions()
    o.intra_op_num_threads=threads
    o.inter_op_num_threads=1
    o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL if enabled else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    o.enable_cpu_mem_arena=arena
    o.enable_mem_pattern=pattern
    o.optimized_model_filepath=str(OUT/(name+'.onnx'))
    if profile:
        o.enable_profiling=True
        o.profile_file_prefix=str(OUT/(name+'-trace'))
    return ort.InferenceSession(str(path),o,providers=['CPUExecutionProvider'])
def measure(s):
    for _ in range(30): s.run(None,feeds)
    samples=[]
    cpu=time.process_time_ns()
    wall=time.perf_counter_ns()
    for _ in range(1000):
        start=time.perf_counter_ns()
        s.run(None,feeds)
        samples.append((time.perf_counter_ns()-start)/1000)
    return {'median_us':statistics.median(samples),'p95_us':float(np.percentile(samples,95)), 'cpu_wall':(time.process_time_ns()-cpu)/(time.perf_counter_ns()-wall)}
configs={'disabled':{},'enabled':{'enabled':True},'no_arena':{'arena':False},'no_pattern':{'pattern':False},'no_memory_reuse':{'arena':False,'pattern':False},'two_threads':{'threads':2},'four_threads':{'threads':4}}
# No timings are collected concurrently. Reverse ordering on alternating repetitions.
rows=[]
for repeat in range(7):
    names=list(configs)
    if repeat%2: names.reverse()
    for name in names:
        s=session(name,**configs[name])
        rows.append({'repeat':repeat+1,'name':name,**measure(s)})
        del s
profiles={}
for name in ('disabled','enabled'):
    s=session(name,profile=True,**configs[name])
    for _ in range(110): s.run(None,feeds)
    trace=Path(s.end_profiling())
    events=json.loads(trace.read_text())
    groups=collections.defaultdict(list)
    for e in events:
        if e.get('cat')=='Node' and e.get('name','').endswith('_kernel_time'):
            groups[e['name']].append(e)
    profiles[name]=[{'name':n,'op':es[0]['args'].get('op_name'),'median_us':statistics.median(e['dur'] for e in es[10:]),'mean_us':statistics.mean(e['dur'] for e in es[10:]),'input':es[0]['args'].get('input_type_shape'),'output':es[0]['args'].get('output_type_shape'),'thread_scheduling_stats':es[0]['args'].get('thread_scheduling_stats')} for n,es in groups.items()]
graphs={}
for name in ('disabled','enabled'):
    m=onnx.load(OUT/(name+'.onnx'))
    graphs[name]=[{'name':n.name,'op':n.op_type,'domain':n.domain,'inputs':list(n.input),'outputs':list(n.output)} for n in m.graph.node]
report={'timings':rows,'profiles':profiles,'graphs':graphs}
(OUT/'ort-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({name:{'median_us':statistics.median(r['median_us'] for r in rows if r['name']==name),'range_us':[min(r['median_us'] for r in rows if r['name']==name),max(r['median_us'] for r in rows if r['name']==name)]} for name in configs},indent=2))
print(json.dumps(profiles,indent=2))
