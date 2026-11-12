"""Coarse full-graph ORT node timings, separate from the fixed-policy comparison."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'.cache/compare-runtime/pass15'
feeds = {'Input3': ((np.arange(784,dtype=np.int32)%256).astype(np.float32)/np.float32(255)).reshape(1,1,28,28)}
repetitions = []
for repeat in range(3):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_profiling = True
    options.profile_file_prefix = str(OUT/f'ort-profile-{repeat+1}')
    session = ort.InferenceSession(str(ROOT/'.cache/compare-runtime/mnist-ir8-opset13.onnx'),
                                 options,providers=['CPUExecutionProvider'])
    for _ in range(1010):
        session.run(None,feeds)
    events = json.loads(Path(session.end_profiling()).read_text())
    nodes = defaultdict(list)
    for event in events:
        if event.get('cat')=='Node' and event['name'].endswith('_kernel_time'):
            nodes[event['name']].append(event['dur'])
    repetitions.append({name:values[10:] for name,values in nodes.items()})
    assert all(len(values)==1000 for values in repetitions[-1].values())
report = dict(ort_version=ort.__version__,repetitions=repetitions,
              configuration=dict(execution='ORT_SEQUENTIAL',intra_op_threads=1,inter_op_threads=1,
                  optimization='ORT_DISABLE_ALL',provider='CPUExecutionProvider',
                  cpu_arena=True,memory_pattern=True,warmups=10,timed_calls=1000,
                  warning='Integer-microsecond profiled nodes and unprofiled Go prepared calls have different scopes.'))
(OUT/'ort-node-samples.json').write_text(json.dumps(report,indent=2)+'\n')
print([{name:statistics.mean(values) for name,values in nodes.items()} for nodes in repetitions])
