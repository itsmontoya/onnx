# Implementation queue

The library completion record and published handoffs live in [NEEDS.md](NEEDS.md);
runtime task definitions and completion checkboxes live in [ROADMAP.md](ROADMAP.md).
Keep status there so this file does not become a second, conflicting backlog.
The backend baseline is `neuralnetwork v0.7.0`: NN-01–NN-12 are complete and
RT-01–RT-14 are complete. The [NN-13 runtime handoff](docs/handoff-nn13.md)
supplies its baseline inputs. CNN adapters and explicit external weights are
implemented; resource/lifecycle hardening (RT-15) is next.

## Start here

1. **RT-01 is complete**: read the [runtime contract](docs/runtime-contract.md)
   and [capability manifest](capabilities.json). They resolve the ownership,
   resource/cancellation, and numerical constraints against backend v0.7.0.
2. **RT-02–RT-09 are complete**: read their handoffs for module verification,
   reproducible wire bindings, immutable tensors, shared inline decoding, and
   validated named graphs, versioned immutable execution plans, serial sessions,
   and M1 numerical adapters.
3. **RT-10 is complete**: read the offline conformance handoff and
   [fixture/provenance guide](testdata/conformance/README.md), including the
   exact upstream-version audit and explicit unsupported reference records.
4. **RT-11 is complete**: run the [MLP and branching examples](docs/getting-started.md)
   and read the [integration report](docs/integration-rt11.md), including the
   four reference agreements at batches 1 and 3 and current M1 restrictions.
5. **RT-12 is complete**: read the [M2 fixtures](testdata/m2/README.md) and
   [NN-13 handoff](docs/handoff-nn13.md).
6. **RT-13 is complete**: read the [CNN fixtures](testdata/m3/README.md) and
   [integration report](docs/integration-rt13.md).
7. **RT-14 is complete**: read the [external fixtures](testdata/external/README.md)
   and [integration report](docs/integration-rt14.md). Complete **RT-15** next.

## Subsequent milestones

| Milestone | Library work | Runtime work |
| --- | --- | --- |
| M2: shape processing | NN-07–NN-08 available in v0.7.0 | RT-12 complete |
| M3: 2D CNNs | NN-09–NN-11 available in v0.7.0 | RT-13 complete |
| M4: external weights and release readiness | Released copying backend | RT-14–RT-15, RT-18 |
| Later: performance and broader coverage | NN-12 available; resume NN-13 after RT-12 | RT-16–RT-17, RT-19 |

The detailed tasks define the actual dependency edges. Optional performance and
version expansion do not block the first working runtime or M4 release readiness.
NN-13 needs an accessible RT-12 revision, MLP and shape-heavy fixtures, named
inputs/expected outputs, and at least two batch sizes. It does not wait for RT-16;
its measurements and optimization feed that later runtime task.

## Every task handoff

Read [AGENTS.md](AGENTS.md), [STYLEGUIDE.md](STYLEGUIDE.md), the relevant plan,
current source, and preceding handoffs. Record changed files, exact APIs and
behavior, dependency revisions, commands/results, and known limitations. Check a
task only after its implementation and acceptance checks pass. Use the target
repository's rules for library work and preserve
[the PR template](.github/PULL_REQUEST_TEMPLATE.md) when preparing a description.
