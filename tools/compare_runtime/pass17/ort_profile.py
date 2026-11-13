"""Run the established coarse ORT diagnostic with a new evidence directory."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE.parent / 'pass15/ort_profile.py').read_text()
source = source.replace("__file__", repr(str(HERE.parent / 'pass15/ort_profile.py')))
source = source.replace("'.cache/compare-runtime/pass15'", "'.cache/compare-runtime/pass17'")
exec(compile(source, str(HERE / 'ort_profile.py'), 'exec'))
