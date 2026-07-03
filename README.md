# motrics

An extremely fast MOT and HOTA metrics library, written in Rust.

`motrics` computes Multiple Object Tracking (MOT) evaluation metrics — CLEAR
(MOTA/MOTP), Identity (IDF1), and HOTA — with a Rust core and ergonomic Python
bindings.

> **Status:** the core metric families — CLEAR, Identity, and HOTA — are
> implemented on a shared IoU + assignment layer, with MOTChallenge ingest and
> numeric parity tests against [TrackEval](https://github.com/JonathonLuiten/TrackEval).
> See the roadmap below.

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

### Benchmarks

`benchmarks/` checks numeric parity and measures speed against
[TrackEval](https://github.com/JonathonLuiten/TrackEval) and
[py-motmetrics](https://github.com/cheind/py-motmetrics):

```bash
uv sync --group parity && uv run maturin develop --release --uv
uv run python benchmarks/download.py     # fetch real MOT17-train sequences
uv run python benchmarks/benchmark.py
```

Roughly how much faster motrics is on real MOT17 (release build; illustrative,
machine-dependent):

| motrics vs…   | CLEAR + Identity | With HOTA |
| ------------- | ---------------- | --------- |
| TrackEval     | ~3–4×            | ~6×       |
| py-motmetrics | ~14×             | —         |

See [`benchmarks/README.md`](benchmarks/README.md) for how to run it.

## Roadmap

- [x] Project scaffolding (build, lint, packaging, CI)
- [x] Bounding-box IoU + assignment (Hungarian/greedy) primitives
- [x] CLEAR metrics (MOTA, MOTP, ID switches, FP/FN)
- [x] Identity metrics (IDF1 / IDP / IDR)
- [x] HOTA (DetA, AssA, alpha sweep)
- [x] MOTChallenge ingest + integration tests
- [x] TrackEval numeric parity tests (CLEAR / Identity / HOTA)
- [x] Benchmark & parity infrastructure vs **TrackEval** and **py-motmetrics**,
      on real MOTChallenge data, validated in CI. See
      [`benchmarks/README.md`](benchmarks/README.md) for methodology, numbers,
      and caveats.
  - [ ] Zero-copy NumPy input path (folds into "broaden core inputs" below).
- [ ] Replace TrackEval / py-motmetrics, not just benchmark against them:
  - [ ] Migration guide + metric-name map.
  - [ ] `motrics.compat.motmetrics` — a drop-in `MOTAccumulator` replacement
        (small surface, no motmetrics installed).
  - [ ] `motrics.compat.trackeval` — a drop-in for the MOTChallenge evaluation
        path (`Evaluator`/dataset/metrics), no TrackEval installed. Gated on
        TrackEval-parity MOTChallenge preprocessing below.
  - [ ] Broaden core inputs (precomputed similarity matrices, `xywh` boxes,
        zero-copy NumPy) so users pass what they already hold.
  - [ ] MOTChallenge ingest with TrackEval-parity preprocessing (confidence,
        class/distractor/ignore-region handling) — the enabling piece for
        `compat.trackeval` and for numbers matching TrackEval's reported values.
- [ ] Pluggable dataset-adapter layer — one metric core, one small adapter per
      benchmark (ingest + preprocessing + similarity), added incrementally:
  - [ ] Box-IoU adapters (DanceTrack, KITTI 2D-box, …) — reuse the existing IoU
        kernel; each is a bounded, low-risk addition with its own parity test.
  - [ ] Mask-IoU similarity kernel (KITTI-MOTS, BDD-MOTS, DAVIS) — new Rust core
        work, not just an adapter; tackle when a mask benchmark is needed.
  - [ ] 3D similarity kernel (KITTI-3D) — same as above, separate core work.

## License

[MIT](LICENSE) © 2026 Kevin Serrano
