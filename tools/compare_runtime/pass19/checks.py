"""Validate the ONNX-only production change using established suite scopes."""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass19', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
spec = importlib.util.spec_from_file_location('checks18', HERE.parent / 'pass18/checks.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
c.p, c.m = p, p.m

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['suites', 'metal', 'metal_host'])
    getattr(c, parser.parse_args().phase)()
