"""Bound machinery using the same traversal with operator bodies disabled."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('diagnostics', HERE / 'diagnostics.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
m = d.m

if __name__ == '__main__':
    variants = ['machinery-control', 'no-operators', 'destinations-no-operators']
    for variant in variants:
        d.build(variant)
    for repeat in range(1, 8):
        for variant in variants if repeat % 2 else list(reversed(variants)):
            m.bench(f'{variant}-diagnostic-{repeat}', variant, '^BenchmarkPass17Run$', '300ms')
