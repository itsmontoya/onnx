# Implementation queue

The library completion record and published handoffs live in [NEEDS.md](NEEDS.md);
runtime task definitions and completion checkboxes live in [ROADMAP.md](ROADMAP.md).
Keep status there so this file does not become a second, conflicting backlog.
The runtime pins the public measured NN-13 backend revision in go.mod.
NN-01–NN-13 and all original RT-01–RT-19 tasks are implemented. The original
19 runtime PRs are merged into the working main baseline; NN-13 and RT-16
are separate optimization follow-ups. Read the
[RT-16 handoff](docs/handoff-rt16.md) and [measurements](docs/performance/rt16/README.md)
for bounded call-local reuse, exact source versions, results and tradeoffs.
RT-19's [future contracts](docs/model-compatibility.md#bounded-next-initiatives)
remain separate work; implementation of an existing kernel does not enable it.

## Start here

1. **RT-01 is complete**: read the [runtime contract](docs/runtime-contract.md)
   and [capability manifest](capabilities.json). They resolve the ownership,
   resource/cancellation, and numerical constraints against the pinned backend.
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
   and [integration report](docs/integration-rt14.md).
8. **RT-15 is complete**: read the [resource/lifecycle audit](docs/hardening-rt15.md).
9. **RT-18 is complete**: read [usage](docs/getting-started.md), the
   [release checklist](docs/release-checklist.md), [notices](third_party/README.md)
   and [handoff](docs/handoff-rt18.md). **RT-16 is complete**; read its
   [storage report](docs/performance/rt16/README.md) and [handoff](docs/handoff-rt16.md). RT-17 is complete; read its [profile handoff](docs/handoff-rt17.md). RT-19 is complete; read its [corpus/normalization handoff](docs/handoff-rt19.md)
   and [bounded future initiatives](docs/model-compatibility.md#bounded-next-initiatives). Publishing a tag and
   deciding the absent project license remain maintainer actions.

## Subsequent milestones

| Milestone | Library work | Runtime work |
| --- | --- | --- |
| M2: shape processing | NN-07–NN-08 available in v0.7.0 | RT-12 complete |
| M3: 2D CNNs | NN-09–NN-11 available in v0.7.0 | RT-13 complete |
| M4: external weights and release readiness | Released copying backend | RT-14–RT-15, RT-18 complete |
| Later: performance and broader coverage | NN-12–NN-13 available at public revisions | RT-16/RT-17/RT-19 complete; unselected families separately scoped |

The detailed tasks define the actual dependency edges. Optional performance and
version expansion do not block the first working runtime or M4 release readiness.
NN-13 used the accessible RT-12 graphs and two-batch fixtures; RT-16 now consumes
its published results. Neither optimization PR publishes a release tag.

## Every task handoff

Read [AGENTS.md](AGENTS.md), [STYLEGUIDE.md](STYLEGUIDE.md), the relevant plan,
current source, and preceding handoffs. Record changed files, exact APIs and
behavior, dependency revisions, commands/results, and known limitations. Check a
task only after its implementation and acceptance checks pass. Use the target
repository's rules for library work and preserve
[the PR template](.github/PULL_REQUEST_TEMPLATE.md) when preparing a description.
