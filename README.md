# motrics

An extremely fast MOT and HOTA metrics library, written in Rust.

`motrics` computes Multiple Object Tracking (MOT) evaluation metrics — CLEAR
(MOTA/MOTP), Identity (IDF1), and HOTA — with a Rust core and ergonomic Python
bindings.

> **Status:** early scaffolding. The build, lint, packaging, and CI pipeline is
> in place; metric implementations are landing incrementally. See the roadmap
> below.

## Install (from source)

```bash
uv sync --group dev        # create the environment + install dev tools
uv run maturin develop     # compile the Rust extension into the venv
uv run python -c "import motrics; print(motrics.version())"
```

## Development

The project is a mixed Rust/Python package built with
[maturin](https://www.maturin.rs/) and [PyO3](https://pyo3.rs/):

```text
src/lib.rs            # Rust core + PyO3 bindings (compiled to motrics._motrics)
python/motrics/       # public Python API surface + type stubs
tests/                # pytest smoke/parity tests
```

Tooling:

| Concern            | Tool                                  |
| ------------------ | ------------------------------------- |
| Build / packaging  | `maturin` (backend), `uv` (env)       |
| Rust format / lint | `cargo fmt`, `cargo clippy`           |
| Python format/lint | `ruff format`, `ruff check`           |
| Python type check  | `ty`                                  |
| Tests              | `cargo test`, `pytest`                |
| CI / release       | GitHub Actions (`.github/workflows/`) |

Common commands:

```bash
cargo fmt --all && cargo clippy --all-targets -- -D warnings
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run maturin develop && uv run pytest
```

Optional: `pre-commit install` to run the formatters/linters on every commit.

## Roadmap

- [x] Project scaffolding (build, lint, packaging, CI)
- [x] Bounding-box IoU + assignment (Hungarian/greedy) primitives
- [x] CLEAR metrics (MOTA, MOTP, ID switches, FP/FN)
- [x] Identity metrics (IDF1 / IDP / IDR)
- [x] HOTA (DetA, AssA, alpha sweep)
- [ ] MOTChallenge / TrackEval ingest + parity tests

## License

[MIT](LICENSE) © 2026 Kevin Serrano
