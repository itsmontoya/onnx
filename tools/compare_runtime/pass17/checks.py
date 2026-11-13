"""Validate unchanged production behavior and the limited diagnostic fixtures."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pass17', HERE / 'measure.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m

if __name__ == '__main__':
    for name, options, extra in [
        ('default', [], {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}),
        ('no-cgo', [], {}), ('purego', ['-tags=purego'], {}),
        ('race', ['-race'], {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}),
        ('purego-race', ['-race', '-tags=purego'], {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}),
    ]:
        m.command('test-' + name, ['go', 'test', '-count=1', *options, './...'], extra)
        for variant in ['current-control', 'destinations-resolved', 'destinations-resolved-no-publication', 'arguments']:
            m.command('diagnostic-' + variant + '-' + name,
                ['go', 'test', '-count=1', '-overlay=' + str(m.OUT / (variant + '-overlay.json')),
                 *options, '-run=^TestPass17Repeat$', '.'], extra)
    m.command('vet', ['go', 'vet', './...'])
    m.command('verify', ['go', 'mod', 'verify'])
    m.command('tidy', ['go', 'mod', 'tidy', '-diff'])
    for target in ['darwin/arm64', 'darwin/amd64', 'linux/arm64', 'linux/amd64', 'windows/amd64', 'linux/386']:
        goos, goarch = target.split('/')
        for tags in ['', 'purego']:
            m.command('build-' + goos + '-' + goarch + ('-purego' if tags else ''),
                ['go', 'build', '-tags=' + tags, './...'], {'GOOS': goos, 'GOARCH': goarch})
    m.command('ort-profile', [m.PYTHON, HERE / 'ort_profile.py'])
    m.command('purego-comparison', [m.PYTHON, 'tools/compare_runtime.py', '--report', m.OUT / 'purego-comparison.json'], {'GOFLAGS': '-tags=purego'})
    for name, cwd in [('onnx', p.ROOT), ('backend', m.BACKEND)]:
        m.command(name + '-metal-serialized', ['go', 'test', '-count=1', '-p=1', '-tags=metal', './...'],
            {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}, cwd)
    m.command('backend-metal-device-detail', ['go', 'test', '-count=1', '-v', '-tags=metal', './matrix', './internal/device'],
        {'CGO_ENABLED': '1', 'GOMAXPROCS': '2'}, m.BACKEND)
