"""Collect fresh coarse node signals with the unchanged ORT configuration."""
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
source = (HERE.parent / 'pass15/ort_profile.py').read_text()
source = source.replace('__file__', repr(str(HERE.parent / 'pass15/ort_profile.py')))
output = Path(os.environ.get('ONNX_PASS19_OUT', HERE.parents[2] / '.cache/compare-runtime/pass19')).resolve()
source = source.replace("ROOT/'.cache/compare-runtime/pass15'", repr(str(output)))
# The inherited probe uses OUT as a Path when constructing profile filenames.
source = source.replace('OUT = ' + repr(str(output)), 'OUT = Path(' + repr(str(output)) + ')')
exec(compile(source, str(HERE / 'ort_profile.py'), 'exec'))
