# motrics

An extremely fast MOT and HOTA metrics library, written in Rust.

`motrics` computes Multiple Object Tracking (MOT) evaluation metrics — CLEAR
(MOTA/MOTP), Identity (IDF1), and HOTA — with a Rust core and ergonomic Python
bindings.

> **Status:** the core metric families — CLEAR, Identity, and HOTA — are
> implemented on a shared IoU + assignment layer, with MOTChallenge ingest.
> Bit-exact TrackEval parity on full benchmark sequences is the remaining
> validation work. See the roadmap below.

## Install (from source)

```bash
uv sync --group dev        # create the environment + install dev tools
uv run maturin develop     # compile the Rust extension into the venv
uv run python -c "import motrics; print(motrics.version())"
```

## Usage

```python
import motrics

# Parse MOTChallenge ground truth and tracker results.
gt = motrics.load_motchallenge("seq/gt/gt.txt")
pred = motrics.load_motchallenge("seq/res.txt", min_confidence=0.5)

# Align onto a shared frame timeline, then compute metrics.
gt_ids, gt_boxes, pred_ids, pred_boxes = motrics.align_frames(gt, pred)

clear = motrics.compute_clear(gt_ids, gt_boxes, pred_ids, pred_boxes)
identity = motrics.compute_identity(gt_ids, gt_boxes, pred_ids, pred_boxes)
hota = motrics.compute_hota(gt_ids, gt_boxes, pred_ids, pred_boxes)

print(clear.mota, identity.idf1, hota.hota)
```

Boxes use the `xyxy` convention `(x1, y1, x2, y2)`. The lower-level primitives
`iou`, `iou_matrix`, and `match_boxes` are also exposed.

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
- [x] MOTChallenge ingest + integration tests
- [ ] Full TrackEval numeric parity on benchmark sequences

## License

[MIT](LICENSE) © 2026 Kevin Serrano
