"""Validate the production lifecycle change, including full build-tag suites."""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass18', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m


def suites():
    for name, options, extra in [
        ('default', [], {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}),
        ('no-cgo', [], {}), ('purego', ['-tags=purego'], {}),
        ('race', ['-race'], {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}),
        ('purego-race', ['-race', '-tags=purego'], {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}),
    ]:
        m.command('test-' + name, ['go', 'test', '-count=1', *options, './...'], extra)
    m.command('vet', ['go', 'vet', './...'])
    m.command('verify', ['go', 'mod', 'verify'])
    m.command('tidy', ['go', 'mod', 'tidy', '-diff'])
    m.command('lifecycle-details', ['go', 'test', '-count=1', '-v', '-run=TestDestinationStorage', '.'])
    for target in ['darwin/arm64', 'darwin/amd64', 'linux/arm64', 'linux/amd64', 'windows/amd64', 'linux/386']:
        goos, goarch = target.split('/')
        for tags in ['', 'purego']:
            m.command('build-' + goos + '-' + goarch + ('-purego' if tags else ''),
                ['go', 'build', '-tags=' + tags, './...'], {'GOOS': goos, 'GOARCH': goarch})
    for package, target in [('./internal/decode', 'FuzzProtobufEntrypoints'),
                           ('./internal/decode', 'FuzzTensorDecode'),
                           ('./internal/graph', 'FuzzGraphReferences'), ('.', 'FuzzShapeAttributePlanning')]:
        m.command('fuzz-' + target, ['go', 'test', package, '-run=^$',
            '-fuzz=^' + target + '$', '-fuzztime=3s', '-parallel=2'], {'GOMAXPROCS': '2'})
    m.command('purego-comparison', [m.PYTHON, 'tools/compare_runtime.py', '--report',
        m.OUT / 'purego-comparison.json'], {'GOFLAGS': '-tags=purego'})


def metal():
    extra = {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}
    for name, cwd in [('onnx', p.ROOT), ('backend', m.BACKEND)]:
        m.command(name + '-metal-sandbox-serialized',
            ['go', 'test', '-count=1', '-p=1', '-v', '-tags=metal', './...'], extra, cwd)


def metal_host():
    extra = {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}
    for name, cwd in [('onnx', p.ROOT), ('backend', m.BACKEND)]:
        m.command(name + '-metal-host-serialized',
            ['go', 'test', '-count=1', '-p=1', '-v', '-tags=metal', './...'], extra, cwd)
    m.command('backend-metal-host-stress', ['go', 'test', '-count=5', '-v', '-tags=metal',
        '-run=^Test_SequentialResidentLongRunningMixedStress$', './model'], extra, m.BACKEND)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['suites', 'metal', 'metal_host'])
    globals()[parser.parse_args().phase]()
