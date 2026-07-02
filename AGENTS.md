# AGENTS.md

Guidance for AI agents and human contributors working in this repository.

## Project

`motrics` is an extremely fast MOT (Multiple Object Tracking) and HOTA metrics
library: a Rust core with Python bindings. It computes tracking-evaluation
metrics — CLEAR (MOTA/MOTP), Identity (IDF1), and HOTA.

It is a **mixed Rust/Python package** built with
[maturin](https://www.maturin.rs/) + [PyO3](https://pyo3.rs/). The compiled
extension is exposed as the private module `motrics._motrics` and re-exported
from the public `motrics` Python package (the Polars / pydantic-core
convention).

## Layout

```text
src/lib.rs            # Rust core + PyO3 bindings -> compiled to motrics._motrics
python/motrics/       # public Python API surface
  __init__.py         #   re-exports from the compiled module
  _motrics.pyi        #   type stubs for the compiled module
  py.typed            #   PEP 561 marker
tests/                # pytest tests (import the built extension)
Cargo.toml            # Rust crate + dependencies
pyproject.toml        # Python metadata + maturin/uv config
ruff.toml ty.toml     # Python lint/format + type-check config
rustfmt.toml          # Rust format config
.github/workflows/    # ci.yml (lint+test), release.yml (wheels + PyPI)
```

Tests live **outside** the package and always run against the built extension.

## Environment & tooling

Python tooling uses the Astral stack via `uv`. Rust uses the standard toolchain.

```bash
uv sync --group dev        # create the venv + install dev tools (ruff, ty, pytest, maturin)
uv run maturin develop     # compile the Rust extension into the venv (rerun after editing Rust)
```

| Concern            | Tool                          |
| ------------------ | ----------------------------- |
| Build / packaging  | `maturin` (backend) + `uv`    |
| Rust format / lint | `cargo fmt`, `cargo clippy`   |
| Python format/lint | `ruff format`, `ruff check`   |
| Python type check  | `ty`                          |
| Tests              | `cargo test`, `pytest`        |

## Checks — run before committing

These mirror CI; all must pass.

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
uv run ruff check .
uv run ruff format --check .
uv run maturin develop --uv    # rebuild so pytest/ty see current Rust
uv run ty check
uv run pytest
```

## Conventions

- **Minimum Python: 3.10.** Wheels are `abi3-py310` (one wheel covers 3.10+).
- **Version** is sourced from `Cargo.toml` — bump it there, not in
  `pyproject.toml` (which declares `version` as dynamic).
- **Prefer well-maintained existing libraries over re-implementing
  algorithms.** Before hand-rolling a non-trivial algorithm (assignment
  solvers, graph/geometry routines, numerical kernels, ...), check crates.io
  for an established crate. Vet it first — recent activity, downloads, a
  compatible license (MIT/Apache-2.0), and a sane dependency tail — and prefer
  one that matches the reference implementation we validate against (e.g. we
  use `lsap`, a port of SciPy's `linear_sum_assignment`, for optimal matching).
  Only re-implement when nothing suitable exists, the dependency cost clearly
  outweighs the benefit (e.g. IoU is ~15 lines and needs no crate), or a crate
  is unmaintained/incompatible — and say so in the PR.
- New Rust functions exposed to Python must also get an entry in
  `python/motrics/_motrics.pyi` and be re-exported from `__init__.py`.
- Follow existing style: Rust idioms per `clippy`, Python per `ruff` + `ty`.
  Annotate every Python function signature.
- **Commits:** Conventional Commits (`feat:`, `fix:`, `chore:`, `ci:`, ...).
- **Branches:** conventional prefixes (`feat/...`, `fix/...`, `chore/...`).
- Keep the roadmap in `README.md` up to date as metrics land.

## Roadmap (implementation order)

1. Bounding-box IoU + assignment (Hungarian / greedy) primitives
2. CLEAR metrics (MOTA, MOTP, ID switches, FP/FN)
3. Identity metrics (IDF1 / IDP / IDR)
4. HOTA (DetA, AssA, alpha sweep)
5. MOTChallenge / TrackEval ingest + parity tests
6. Real-data benchmark & parity — unify the parity and benchmark inputs onto
   shared MOTChallenge-format fixtures, add a reproducible `benchmarks/` suite,
   and validate parity + measure speedups (vs TrackEval, py-motmetrics) on real
   MOTChallenge sequences. Needs network access to fetch the dataset (blocked in
   the sandbox we developed in; available in CI or a permissioned session).
   Note: parity so far uses synthetic in-memory sequences, and the initial
   benchmark used a separate synthetic generator — this step makes both use one
   real (or shared) dataset. Ballpark synthetic result: motrics ~6-8x faster
   than TrackEval end-to-end (~26x vs py-motmetrics); a numpy zero-copy
   fast-path is the likely next perf lever.

When adding a metric, validate numbers against a reference implementation
(e.g. TrackEval) in `tests/`.
