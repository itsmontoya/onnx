#!/usr/bin/env python3
"""Observe exact pinned graphs in temporary instrumented source archives.

This uses the published NN-13 observer only for numeric payload copies and its
conservative function-scoped lifetime bound. Explicit reservation hooks measure
runtime live/workspace high-water marks; these are separate from heap or RSS.
Never use these instrumented copies for timing or Go allocation measurements.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


DEFAULT_BACKEND = "a64fa687e269c3a1bf3ac345434b46c8ed4a59fc"
TOOLING_REVISION = "458b64d18a2e974a1bd51313d040060a85f1f70a"


def run(arguments, **kwargs):
    try:
        return subprocess.run(arguments, check=True, text=True, **kwargs)
    except subprocess.CalledProcessError as failure:
        if failure.stdout:
            print(failure.stdout)
        if failure.stderr:
            print(failure.stderr)
        raise


def archive(repository, revision, destination):
    destination.mkdir()
    source = subprocess.Popen(["git", "-C", str(repository), "archive", revision], stdout=subprocess.PIPE)
    try:
        run(["tar", "-x", "-C", str(destination)], stdin=source.stdout)
    finally:
        source.stdout.close()
    if source.wait():
        raise RuntimeError("git archive failed")


def replace_once(path, old, new):
    source = path.read_text()
    if source.count(old) != 1:
        raise RuntimeError(f"pinned observer marker changed in {path.name}")
    path.write_text(source.replace(old, new))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--backend-revision", default=DEFAULT_BACKEND)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.runtime_revision = subprocess.check_output(
        ["git", "-C", str(arguments.runtime), "rev-parse", "--verify", arguments.runtime_revision + "^{commit}"], text=True).strip()
    arguments.backend_revision = subprocess.check_output(
        ["git", "-C", str(arguments.backend), "rev-parse", "--verify", arguments.backend_revision + "^{commit}"], text=True).strip()
    tooling = Path(__file__).resolve().parent
    environment = dict(os.environ, GOMAXPROCS="1", CGO_ENABLED="0", GOWORK="off")
    with tempfile.TemporaryDirectory(prefix="rt16-metrics-") as temporary:
        root = Path(temporary).resolve()
        runtime, backend, observer = root / "onnx", root / "backend", root / "observer"
        archive(arguments.runtime, arguments.runtime_revision, runtime)
        archive(arguments.backend, arguments.backend_revision, backend)
        archive(arguments.backend, TOOLING_REVISION, observer)
        shutil.copytree(observer / "tools/nn13", backend / "tools/nn13", dirs_exist_ok=True)
        trace = backend / "tools/nn13/trace"
        replace_once(trace / "snapshot.go", "type Snapshot struct {", '''type Snapshot struct {
    ReservationPeakBytes int64 `json:"reservation_peak_bytes"`
    WorkspacePeakBytes int64 `json:"workspace_peak_bytes"`
    ReusedDestinations int `json:"reused_destinations"`''')
        with (trace / "trace.go").open("a") as output:
            output.write('''
// Reservation records the actual successful logical reservation high-water mark.
func Reservation(live, workspace int64) {
    if enabled {
        result.ReservationPeakBytes = max(result.ReservationPeakBytes, live+workspace)
        result.WorkspacePeakBytes = max(result.WorkspacePeakBytes, workspace)
    }
}

// Reuse counts successful borrows of a retired private destination.
func Reuse() {
    if enabled {
        result.ReusedDestinations++
    }
}
''')
        binary = root / "instrument"
        run(["go", "build", "-o", str(binary), "./tools/nn13/instrument"], cwd=backend, env=environment)
        run([str(binary), str(backend)])
        run([str(binary), str(runtime)])
        replace_once(runtime / "shape_call.go",
                     "c.shape, c.shapeValues = []int64{int64(end - start)}, left[start:end]",
                     "c.shape, c.shapeValues = []int64{int64(end - start)}, left[start:end]\n nn13trace.Track(c.shapeValues)")
        replace_once(runtime / "run_budget.go", "b.workspace += workspace",
                     "b.workspace += workspace\n nn13trace.Reservation(b.live, b.workspace)")
        storage = runtime / "run_storage.go"
        if storage.exists():
            replace_once(storage, "s.reuses++", "s.reuses++\n nn13trace.Reuse()")
        shutil.copyfile(tooling / "runtime_metrics_test.go.txt", runtime / "rt16_metrics_test.go")
        run(["go", "work", "init", "./onnx", "./backend"], cwd=root, env=environment)
        environment["GOWORK"] = str(root / "go.work")
        result = run(["go", "test", "-run", "^TestRT16PayloadMetrics$", "-count=1", "-v", "."],
                     cwd=runtime, env=environment, capture_output=True)
        observations = [json.loads(line[len("RT16METRIC "):]) for line in result.stdout.splitlines()
                        if line.startswith("RT16METRIC ")]
        if len(observations) != 24:
            raise RuntimeError(f"expected 24 observations, got {len(observations)}\n{result.stdout}")
        observations.sort(key=lambda item: (item["case"], item["phase"]))
        report = {"runtime_revision": arguments.runtime_revision, "backend_revision": arguments.backend_revision,
                  "observer_revision": TOOLING_REVISION, "metrics": observations}
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote 24 graph/phase observations to {arguments.output}")


if __name__ == "__main__":
    main()
