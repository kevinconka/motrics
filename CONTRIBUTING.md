# Contributing

Thanks for considering a contribution to `motrics`. This covers getting set
up and the checks a PR needs to pass; for project layout, coding conventions,
and the full roadmap, see [AGENTS.md](AGENTS.md) — it's written for human and
AI contributors alike.

## Setup

The project is a mixed Rust/Python package built with
[maturin](https://www.maturin.rs/) and [PyO3](https://pyo3.rs/):

```bash
uv sync --group dev        # create the environment + install dev tools
uv run maturin develop     # compile the Rust extension into the venv (rerun after editing Rust)
```

| Concern            | Tool                                  |
| ------------------ | -------------------------------------- |
| Build / packaging  | `maturin` (backend), `uv` (env)       |
| Rust format / lint | `cargo fmt`, `cargo clippy`           |
| Python format/lint | `ruff format`, `ruff check`           |
| Python type check  | `ty`                                  |
| Tests              | `cargo test`, `pytest`                |
| CI / release       | GitHub Actions (`.github/workflows/`) |

Optional: `pre-commit install` to run the formatters/linters on every commit.

## Checks — run before opening a PR

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

## Benchmarks

`benchmarks/` checks numeric parity and measures speed against
[TrackEval](https://github.com/JonathonLuiten/TrackEval) and
[py-motmetrics](https://github.com/cheind/py-motmetrics) on real MOTChallenge
data:

```bash
uv sync --group parity && uv run maturin develop --release --uv
uv run python benchmarks/download.py     # fetch real MOT17-train sequences
uv run python benchmarks/benchmark.py
```

See [`benchmarks/README.md`](benchmarks/README.md) for methodology and
current numbers.

## Conventions

- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `chore:`, `ci:`, ...).
- **Branches:** conventional prefixes (`feat/...`, `fix/...`, `chore/...`).
- When adding a metric, validate numbers against a reference implementation
  (e.g. TrackEval) in `tests/`.
- New Rust functions exposed to Python must also get an entry in
  `python/motrics/_motrics.pyi` and be re-exported from `__init__.py`.
- Keep the roadmap in `README.md` up to date as features land.

See [AGENTS.md](AGENTS.md) for the rest: project layout, the
prefer-a-crate-over-hand-rolling-it policy, and versioning.
