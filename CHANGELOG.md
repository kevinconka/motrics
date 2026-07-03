# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/kevinconka/motrics/releases/tag/v0.1.0) - 2026-07-03

### Added

- add Frames container and evaluate() for CLEAR+Identity+HOTA in one call ([#18](https://github.com/kevinconka/motrics/pull/18))
- broaden core box inputs (xywh format, zero-copy NumPy) ([#17](https://github.com/kevinconka/motrics/pull/17))
- add motrics.compat.trackeval, a TrackEval class-based drop-in ([#16](https://github.com/kevinconka/motrics/pull/16))
- MOTChallenge ingest with TrackEval-parity preprocessing ([#15](https://github.com/kevinconka/motrics/pull/15))
- add motrics.compat.motmetrics, a drop-in MOTAccumulator replacement ([#13](https://github.com/kevinconka/motrics/pull/13))
- add reproducible benchmark suite and unify parity on shared fixtures
- add MOTChallenge ingest and end-to-end integration tests
- add HOTA metrics (DetA, AssA, LocA, alpha sweep)
- add Identity metrics (IDF1, IDP, IDR)
- add CLEAR MOT metrics (MOTA, MOTP, FP/FN, ID switches)

### Fixed

- revert release-plz.toml to publish = false ([#22](https://github.com/kevinconka/motrics/pull/22))
- *(benchmarks)* tolerate assignment tie-breaking in real-data parity gate

### Other

- wire up release-plz and bump to v0.1.0 for the first release ([#20](https://github.com/kevinconka/motrics/pull/20))
- confirm DanceTrack works via existing MOTChallenge ingest ([#19](https://github.com/kevinconka/motrics/pull/19))
- split README (user-facing) from CONTRIBUTING.md (dev setup) ([#14](https://github.com/kevinconka/motrics/pull/14))
- reframe migration as replacement + add pluggable dataset-adapter roadmap
- harden workflows per review (shell-injection + persist-credentials)
- add adoption & migration roadmap (compat shims, input formats, MOT preprocessing)
- rename cryptic 'te' to 'trackeval' in parity tests
- *(benchmarks)* drop import-guard globals and dual-timing machinery
- *(benchmarks)* real-data-only benchmark, single workflow, leaner CI
- *(benchmarks)* simplify fixtures, tests, docs, and CI per review
- add opt-in real-data benchmark workflow
- consolidate jobs and clarify the benchmark results table
- *(benchmarks)* git-ignore generated fixtures and add profiling table
- add real-data benchmark & parity roadmap step
- add TrackEval numeric parity tests for CLEAR/Identity/HOTA
- prefer maintained libraries over re-implementing algorithms
- use the lsap crate for optimal assignment
- Merge branch 'main' into dependabot/github_actions/actions/checkout-7
- Merge branch 'main' into dependabot/github_actions/actions/download-artifact-8
- *(deps)* bump actions/upload-artifact from 4 to 7
- add AGENTS.md and .claude/CLAUDE.md import
- fix duplicate platform tag in linux wheel build
- scaffold Rust/Python project with maturin, tooling, and CI
- Initial commit
