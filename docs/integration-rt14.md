# RT-14 external weight integration report

Verified on 2026-11-08 using Go 1.27.2, Darwin/arm64, from RT-13 base
`00253078fd95fc0314cac252143a3c5efb8d4b6b`. Final implementation revision:
[`f02e7bace484ebc6cb8be67850ca9b85665c991d`](https://github.com/itsmontoya/onnx/tree/f02e7bace484ebc6cb8be67850ca9b85665c991d).
Implementation and fixture commits are respectively
`a60003498c18b6f479d2279f9981ffc0e1bd1597` and
`f8713414ccb9ec3d5480706d6d749fc9fd9206ba`; the final implementation revision
also applies the repository's explicit local-concrete constructor pattern.

Dependencies remain `github.com/itsmontoya/neuralnetwork v0.7.0` at
`6048452c57a0a293db4591cd2e2febd7d6b743a3`, protobuf v1.36.12, toolchain Go
1.27.2 and module language baseline Go 1.26.1. No backend, local replacement,
module, operator, layout or model-envelope expansion is introduced. IR 8,
empty standard domain and imported opset exactly 13 remain required.

## APIs and ownership

`Options.ExternalData` now accepts the existing explicit interface:

```go
type ExternalDataResolver interface {
    Resolve(ctx context.Context, location string, offset, length int64) (source io.ReadCloser, err error)
}

func NewDirectoryResolver(directory string) (out *DirectoryResolver, err error)
func (r *DirectoryResolver) Resolve(ctx context.Context, location string,
    offset, length int64) (source io.ReadCloser, err error)
func (r *DirectoryResolver) Close() (err error)
```

All public loaders, including LoadFile, require this resolver for external
float32/int64 tensors. Inline-only models accept a resolver without invoking it.
The loader never guesses a directory or automatically fetches a URL. A custom
resolver may explicitly permit an alternate storage source; its implementation
controls the interpretation and authorization of location names.

The resolver remains caller-owned. The loader closes each nonnil returned reader
exactly once, including a reader returned with an error. It reads exactly the
validated tensor bytes in at most 32 KiB chunks, handles partial reads and bounds
no-progress loops, checks cancellation, and closes before converting. A short
read yields io.ErrUnexpectedEOF; resolution, read and close errors retain their
causes and tensor/location context. Joined failures preserve both causes. Close
failure prevents publication. Zero-byte ranges resolve and close without Read.
Blocked caller Resolve/Read calls require that implementation's cancellation
cooperation; synchronous backend copies are bracketed but not interruptible.

Typed/raw conversion is shared through one complete-model TensorDecoder. Its
consumer-side minimal resolver interface avoids a public/internal import cycle.
A narrow fresh TensorProto view carries dimensions and runtime-owned raw bytes
into the existing conversion path; generated messages containing locks are not
copied. External read bytes, typed staging and owned tensor storage reserve
three payload widths against aggregate MaxLoadBytes before I/O/allocation.
Owned bytes also count against aggregate MaxInitializerBytes and each tensor
against MaxTensorBytes. Successful staging reservations remain conservative;
failed or cancelled conversion commits none. Reused locations and ranges produce
separate owned values. Caller buffers, external resources, later loads/runs,
and resolver/model/session Close cannot invalidate published values.

## Metadata and directory confinement

External tensors require EXTERNAL data_location and a nonempty location. Offset
and supplied length accept only nonnegative decimal int64 strings. A supplied
length must equal the exact shape/type byte count; omitted length still resolves
that count. Range addition is checked. Nil/duplicate entries, malformed numbers,
wrong types/shapes/segments and external/inline encoding conflicts are rejected
before numerical allocation or I/O for that tensor. Unknown keys, including
basepath, are unsupported.

The pinned [ONNX v1.16.2 specification](https://github.com/onnx/onnx/blob/3bf92c03a9f27eba3bda1e5b9e63ea20ec213557/docs/ExternalData.md)
and [vendored schema](../third_party/onnx/onnx.proto) define optional checksum as
whole-file SHA-1. Checksum metadata is explicitly unsupported until verified
whole-file policy is implemented. Fixture SHA-256 provenance does not implement
or claim ONNX checksum verification.

DirectoryResolver uses os.OpenRoot and Root.Open. Confinement is checked during
open and does not use EvalSymlinks followed by an unprotected Open. It rejects
absolute paths, traversal components, backslashes, drive/URL forms and symlink
escapes. Internal relative symlinks are accepted. The opened file must be regular
and its requested range must fit the opened file's current size. Each resolve
opens an independent bounded reader, without shared mutable offsets.

Resolve and Close are safe concurrently. Close is idempotent and prevents new
resolves with ErrClosed, while existing readers remain usable and independently
closable. Nil/zero helpers are invalid. File contents may still change or truncate
while read; callers requiring stable bytes must arrange that stability.

The installed official Go os.Root documentation/source supplies its race-safe
open contract. Its documented js symlink race and plan9 rename limitation mean
the helper returns ErrUnsupported on those platforms; callers must supply a
resolver. Other platform behavior follows os.Root; no additional filesystem
sandbox is claimed for hard links, mounts, filesystem boundaries or special
files. Local confinement and racing-replacement tests ran on Darwin/arm64.
Linux/amd64 is covered by the hosted workflow; other platforms were not executed
locally. Symlink tests explicitly skip if the host cannot create symlinks.

## Fixtures and validation

[The external MLP](../testdata/external/README.md) retains the RT-11 MLP graph and
identical weight values, placing four float32 tensors in one shared external
file after a 13-byte prefix. Every offset is nonzero and the last length is
omitted. The actual pinned ONNX 1.16.2 checker, external-data loader and reference
evaluator accept it and reproduce inherited batch 1 and 3 outputs exactly.
The [portable example](../examples/external/main.go) uses an immutable memory
resolver that explicitly allows weights.bin, without os.Root assumptions.

| Artifact | SHA-256 |
| --- | --- |
| External MLP model | `2839759a10df8272f70ee1b633128804ce065187fe9e8fa3a55b286ff80e17ea` |
| External shared weight file | `44cc639beb769ce884d9f807a3802bde3e0295138d31695d98d9d56943138925` |
| External provenance | `d311eaceb990ff06dd1914063e08350a77d5171497ea29be00865c6024162d08` |
| Unchanged inline inference provenance | `547933bf4e3be8e5e57e537820a1529822fcf3ddda17d953a83e98eca887ff2c` |
| Unchanged M1 provenance | `19227f9920883b065788cc8d1a2acdfa803c632a8bc0b4bc24ff720157d315ce` |
| Unchanged M2 provenance | `4bd43bb170feb5427a4f596ba4a171ff1da03cd251558e80e22adb1121fba59f` |
| Unchanged M3 provenance | `be7505137619c7ea8402eb3573bf678788b7c97b88c2ac1c6cfe12b16667ecfb` |
| Unchanged conformance provenance | `bd1b7d7d8659ef7289207d33772d0651226f5df7cc5b6639a4800a404e69849e` |

Provenance records every selected weight range and all named input/output hashes.
All inherited model/value/provenance bytes remain unchanged. Python 3.9.6,
ONNX 1.16.2 at source `3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`, NumPy
1.26.4 and protobuf 4.25.3 reproduce the new fixture. These are development-only
tools; ordinary Go tests stay offline and Python-free. The repository has no
LICENSE; no license was invented for original fixtures.

Passed formatting, tidy-diff, module verification, build, vet, normal tests,
CGO_ENABLED=0, CGO_ENABLED=0 purego and race tests using
`GOCACHE=/private/tmp/onnx-runtime-gocache` and cached offline dependencies.
The external example passes batches 1 and 3 plus purego batch 3. A clean local
implementation clone passes cached module download, cgo-disabled tests and those
three examples with a clean tree. All protobuf/profile/M1/M2/M3/conformance/
inference/external generators reproduce. Changed handwritten files satisfy
if/var spacing and type/method colocation; local Markdown targets and diff
checks pass.

Tests cover missing/duplicate/unknown/checksum metadata, malformed decimal and
range overflow, unsupported type/segment/encoding conflicts before I/O; aggregate
external and initializer/Constant budgets; scalar/empty tensors, omitted lengths,
float bits and int64 precision; bounded partial/exact/short/no-progress/error
reads; joined resolve/read/close failures; cancellation before reads, between
chunks and during conversion; independent/shared ranges and locations; all three
loaders, owned data after caller-buffer mutation and Close; concurrent loads and
sessions; traversal/symlink escape and concurrent path replacement.

The next runtime task is RT-15 resource/error/cancellation/lifecycle hardening.
Further profile expansion, checksum verification, optimization and release
readiness remain separate. Keep the RT-14 PR open and unmerged on RT-13; no later
task is started by this change.
