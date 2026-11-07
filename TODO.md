# Implementation queue

The library completion record and published handoffs live in [NEEDS.md](NEEDS.md);
runtime task definitions and completion checkboxes live in [ROADMAP.md](ROADMAP.md).
Keep status there so this file does not become a second, conflicting backlog.
The backend baseline is `neuralnetwork v0.7.0`: NN-01–NN-12 are complete and
NN-13 is blocked on RT-12. RT-01–RT-08 are complete; M1 numerical adapters
(RT-09) are next.

## Start here

1. **RT-01 is complete**: read the [runtime contract](docs/runtime-contract.md)
   and [capability manifest](capabilities.json). They resolve the ownership,
   resource/cancellation, and numerical constraints against backend v0.7.0.
2. **RT-02–RT-08 are complete**: read their handoffs for module verification,
   reproducible wire bindings, immutable tensors, shared inline decoding, and
   validated named graphs, versioned immutable execution plans, and serial sessions.
3. Complete **RT-09** using the released tensor storage, shape helpers,
   and M1 kernels. Build the **RT-10** fixture harness alongside them.
4. Complete **RT-11** to demonstrate M1 with a working MLP and branching graph.
5. Complete **RT-12** and hand off reproducible runtime graphs to unblock NN-13.

## Subsequent milestones

| Milestone | Library work | Runtime work |
| --- | --- | --- |
| M2: shape processing | NN-07–NN-08 available in v0.7.0 | RT-12 |
| M3: 2D CNNs | NN-09–NN-11 available in v0.7.0 | RT-13 |
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
