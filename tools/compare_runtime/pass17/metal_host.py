"""Repeat serialized Metal validation on a host with actual GPU access."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass17', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m

if __name__ == '__main__':
    extra = {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}
    for name, cwd in [('onnx', p.ROOT), ('backend', m.BACKEND)]:
        m.command(name + '-metal-host-serialized', ['go', 'test', '-count=1', '-p=1', '-v', '-tags=metal', './...'], extra, cwd)
    m.command('backend-metal-host-stress', ['go', 'test', '-count=5', '-v', '-tags=metal',
        '-run=^Test_SequentialResidentLongRunningMixedStress$', './model'], extra, m.BACKEND)
