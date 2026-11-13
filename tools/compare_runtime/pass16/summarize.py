"""Package every Pass 16 sample and the validation logs, without test binaries."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / '.cache/compare-runtime/pass16'
DATA = HERE / 'data'


def main():
    rows = {}
    for filename in ['backend-bench.txt', 'onnx-nodes.txt', 'onnx-ramp.txt']:
        for match in re.finditer(
                r'^(Benchmark\S+)\s+\d+\s+([\d.]+) ns/op\s+(\d+) B/op\s+(\d+) allocs/op',
                (OUT / filename).read_text(), re.M):
            name, latency, byte_count, allocations = match.groups()
            rows.setdefault(name, []).append((float(latency) / 1000, int(byte_count), int(allocations)))
    assert len(rows) == 21
    measurements = {}
    for name, values in rows.items():
        assert len(values) == 7
        latencies = [value[0] for value in values]
        measurements[name] = dict(median_us=statistics.median(latencies), min_us=min(latencies),
                                  max_us=max(latencies), cv_pct=100 * statistics.pstdev(latencies) /
                                  statistics.mean(latencies), B_op=sorted(set(value[1] for value in values)),
                                  allocs_op=sorted(set(value[2] for value in values)), samples=values)
    environment = json.loads((OUT / 'environment.json').read_text())
    original_hashes = json.loads((OUT / 'source-sha256.json').read_text())
    # Production source, model and diagnostic source must match the measured build.
    for name, expected in original_hashes.items():
        if name.endswith('measure.py'):
            continue
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    source_hashes = dict(original_hashes)
    for path in HERE.glob('*.py'):
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    commands = [json.loads(line) for line in (OUT / 'commands.jsonl').read_text().splitlines()]
    latest = {record['name']: record for record in commands}
    assert all(record['exit_code'] == 0 for record in latest.values())
    for suffix in ['no-cgo', 'purego', 'race', 'purego-race']:
        assert latest['diagnostic-' + suffix]['exit_code'] == 0
        assert latest['onnx-' + suffix]['exit_code'] == 0
    for name in ['add-alloc_objects', 'add-alloc_space', 'matmul-alloc_objects', 'matmul-alloc_space',
                 'add-sites', 'matmul-sites']:
        assert (OUT / (name + '.txt')).exists()
    logs = {path.name: path.read_text() for path in sorted(OUT.glob('*.txt'))
            if path.name != 'hardware.txt'}
    payload = dict(environment=environment, measurements=measurements, commands=commands,
                   source_sha256=source_hashes, logs=logs)
    DATA.mkdir(exist_ok=True)
    (DATA / 'summary.json').write_text(json.dumps(dict(environment=environment,
        measurements=measurements, source_sha256=source_hashes), indent=2) + '\n')
    (DATA / 'validation.json').write_text(json.dumps(commands, indent=2) + '\n')
    with (DATA / 'evidence.json.gz').open('wb') as output:
        with gzip.GzipFile(fileobj=output, mode='wb', mtime=0) as archive:
            archive.write((json.dumps(payload, separators=(',', ':')) + '\n').encode())
    print('Packaged 21 benchmarks, seven samples each, profiles, escape analysis and validation.')


if __name__ == '__main__':
    main()
