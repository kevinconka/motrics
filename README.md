# motrics

[![CI](https://github.com/kevinconka/motrics/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinconka/motrics/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/kevinconka/motrics)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

An extremely fast MOT and HOTA metrics library, written in Rust — CLEAR
(MOTA/MOTP), Identity (IDF1), and HOTA, with an ergonomic Python API.

## Highlights

- ⚡ **Extremely fast** — Rust core, ~3–14× faster than TrackEval and
  py-motmetrics on real MOT17 data.
- 🎯 **Numerically validated** — exact parity with TrackEval on CLEAR,
  Identity, and HOTA, checked in CI.
- 🔄 **Drop-in migration** — swap one import to replace py-motmetrics; evaluate
  a MOTChallenge benchmark without installing TrackEval.
- 🐍 **Ergonomic, typed Python API** — PEP 561, `numpy` the only required
  runtime dependency.
- 🔢 **Flexible box input** — `xyxy` or `xywh`, and a zero-copy read path for
  contiguous NumPy arrays.

## Install

Not on PyPI yet — build from source:

```bash
uv sync --group dev        # create the environment + install dev tools
uv run maturin develop     # compile the Rust extension into the venv
uv run python -c "import motrics; print(motrics.version())"
```

## Quickstart

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

Boxes default to the `xyxy` convention `(x1, y1, x2, y2)`; pass
`box_format="xywh"` for `(x, y, width, height)` instead. Each frame's boxes
can also be a `(N, 4)` float64 NumPy array — a contiguous `xyxy` array is read
with zero copies, no per-box Python overhead.

Want numbers matching TrackEval's own reported values (pedestrian-only,
distractor-aware)? Use `load_motchallenge_gt` + `preprocess_motchallenge`
instead of `load_motchallenge` + `align_frames`.

## Migrating from py-motmetrics

Swap the import — the rest of your accumulator code is unchanged:

```python
# before
import motmetrics as mm

# after — same code, motrics underneath
import motrics.compat.motmetrics as mm

acc = mm.MOTAccumulator(auto_id=True)
for gt_ids, gt_boxes, pred_ids, pred_boxes in sequence:
    dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
    acc.update(gt_ids, pred_ids, dists)

summary = mm.metrics.create().compute(acc, metrics=mm.metrics.SUPPORTED, name="acc")
```

- `pip install motrics[compat]` (pulls in pandas, needed only for this
  subpackage).
- Supported: `mota`, `motp`, `idf1`, `idp`, `idr`, `recall`, `precision`,
  `num_false_positives`, `num_misses`, `num_switches`, `num_unique_objects` —
  the same names py-motmetrics uses.
- Not yet implemented: per-trajectory metrics (mostly-tracked, fragmentations,
  transfer/ascend/migrate). Requesting them raises `NotImplementedError`
  naming exactly what's missing, rather than a silently wrong number.
- See [`python/motrics/compat/motmetrics/`](python/motrics/compat/motmetrics/)
  for what else differs (e.g. no `events`/`mot_events` DataFrame).

## Migrating from TrackEval

Swap the import — the rest of your evaluation script is unchanged:

```python
# before
import trackeval

# after — same code, motrics underneath
import motrics.compat.trackeval as trackeval

eval_config = trackeval.Evaluator.get_default_eval_config()
evaluator = trackeval.Evaluator(eval_config)

dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dataset_config["GT_FOLDER"] = "data/gt/mot_challenge/"
dataset_config["TRACKERS_FOLDER"] = "data/trackers/mot_challenge/"
dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]

results, messages = evaluator.evaluate(dataset_list, metrics_list)
print(results["MotChallenge2DBox"]["my_tracker"]["COMBINED_SEQ"]["pedestrian"]["CLEAR"]["MOTA"])
```

Same class names, config keys, directory/seqmap conventions
(`GT_FOLDER/BENCHMARK-SPLIT/<seq>/gt/gt.txt`, a seqmap file, per-sequence
`seqinfo.ini`), and result shape as real TrackEval, no `trackeval`/`scipy`
install required — but only for the subset below. `HOTA`, `Identity`, and
`CLEAR`'s `MOTA`/`MOTP` fields are verified bit-exact against real TrackEval;
unsupported config or fields raise rather than silently returning a wrong
number.

- No extra needed — this subpackage relies only on `numpy` (a core
  dependency), whose arrays back HOTA's per-alpha fields, matching TrackEval.
- Not implemented: parallel evaluation, error-handling config
  (`BREAK_ON_ERROR`/etc. — always raises immediately), printing/file
  output/plotting; zipped input (`INPUT_AS_ZIP`); `DO_PREPROC=False` and
  `BENCHMARK="MOT15"` (raise at construction); `CLEAR` fields beyond
  `MOTA`/`MOTP` (`MT`/`PT`/`ML`/`Frag`/`MODA`/`sMOTA`/etc. need
  mostly-tracked/lost and fragmentation bookkeeping the Rust core doesn't
  compute yet); `IDEucl`/`JAndF`/`TrackMAP`/`VACE` metrics.
- See [`python/motrics/compat/trackeval/`](python/motrics/compat/trackeval/)
  for the full list of what differs from real TrackEval.

<details>
<summary>Metric name map — TrackEval / py-motmetrics / motrics' native API</summary>

Using `motrics`' own API directly (faster than the compat layer — no
per-frame Python bookkeeping)? Here's how the field names line up:

| Concept                              | TrackEval                   | py-motmetrics          | `motrics` (native)                         |
| ------------------------------------- | ---------------------------- | ------------------------ | -------------------------------------------- |
| Matched detections (incl. switches)   | `CLR_TP`                    | `num_detections`       | `ClearMetrics.num_matches`                 |
| False positives                       | `CLR_FP`                    | `num_false_positives`  | `ClearMetrics.num_false_positives`         |
| Misses                                | `CLR_FN`                    | `num_misses`           | `ClearMetrics.num_misses`                  |
| Identity switches                     | `IDSW`                      | `num_switches`         | `ClearMetrics.num_switches`                |
| MOTA / MOTP                           | `MOTA` / `MOTP`             | `mota` / `motp`        | `ClearMetrics.mota` / `.motp`              |
| Identity TP / FP / FN                 | `IDTP`/`IDFP`/`IDFN`        | `idtp`/`idfp`/`idfn`   | `IdentityMetrics.idtp`/`.idfp`/`.idfn`     |
| IDF1 / IDP / IDR                      | `IDF1`/`IDP`/`IDR`          | `idf1`/`idp`/`idr`     | `IdentityMetrics.idf1`/`.idp`/`.idr`       |
| HOTA / DetA / AssA / LocA             | `HOTA`/`DetA`/`AssA`/`LocA` | — (not in motmetrics)  | `HotaMetrics.hota`/`.deta`/`.assa`/`.loca` |

</details>

## Benchmarks

On real MOT17 data, release build:

| motrics vs…   | CLEAR + Identity | With HOTA |
| ------------- | ---------------- | --------- |
| TrackEval     | ~3–4×            | ~6×       |
| py-motmetrics | ~14×             | —         |

Numbers are illustrative and machine-dependent. See
[`benchmarks/README.md`](benchmarks/README.md) for methodology and how to run
it yourself.

<details>
<summary>Roadmap</summary>

- [x] Project scaffolding (build, lint, packaging, CI)
- [x] Bounding-box IoU + assignment (Hungarian/greedy) primitives
- [x] CLEAR metrics (MOTA, MOTP, ID switches, FP/FN)
- [x] Identity metrics (IDF1 / IDP / IDR)
- [x] HOTA (DetA, AssA, alpha sweep)
- [x] MOTChallenge ingest + integration tests
- [x] TrackEval numeric parity tests (CLEAR / Identity / HOTA)
- [x] Benchmark & parity infrastructure vs **TrackEval** and **py-motmetrics**,
      on real MOTChallenge data, validated in CI.
  - [x] Zero-copy NumPy input path (see "broaden core inputs" below).
- [ ] Replace TrackEval / py-motmetrics, not just benchmark against them:
  - [x] Precomputed-similarity core inputs (`compute_clear_from_similarity`,
        `compute_identity_from_similarity`) — the piece `compat.motmetrics`
        needed, and the first slice of "broaden core inputs" below.
  - [x] `motrics.compat.motmetrics` — a drop-in `MOTAccumulator` replacement.
  - [x] Migration guide + metric-name map (see above).
  - [x] MOTChallenge ingest with TrackEval-parity preprocessing
        (`load_motchallenge_gt` + `preprocess_motchallenge`: distractor-class
        removal, pedestrian-only, "do not consider" rows dropped) — validated
        against TrackEval's own `get_preprocessed_seq_data`, and now what the
        real-data benchmark uses. The enabling piece for `compat.trackeval`.
  - [x] `motrics.compat.trackeval` — a drop-in for TrackEval's
        `Evaluator`/`datasets.MotChallenge2DBox`/`metrics.{HOTA,CLEAR,Identity}`
        (same class names, config keys, and result shape); see above for
        what's out of scope (parallel eval, full `CLEAR` field set, other
        metrics).
  - [x] Broaden core inputs further — `box_format="xywh"` alongside the
        default `xyxy`, and a zero-copy read path for contiguous `(N, 4)`
        float64 NumPy arrays, on `compute_clear`/`compute_identity`/
        `compute_hota`/`iou_matrix`/`match_boxes`. `numpy` is now the one
        required runtime dependency of the core.
- [ ] Pluggable dataset-adapter layer — one metric core, one small adapter per
      benchmark (ingest + preprocessing + similarity), added incrementally:
  - [ ] Box-IoU adapters (DanceTrack, KITTI 2D-box, …) — reuse the existing IoU
        kernel; each is a bounded, low-risk addition with its own parity test.
  - [ ] Mask-IoU similarity kernel (KITTI-MOTS, BDD-MOTS, DAVIS) — new Rust core
        work, not just an adapter; tackle when a mask benchmark is needed.
  - [ ] 3D similarity kernel (KITTI-3D) — same as above, separate core work.

</details>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, tooling,
and checks to run before opening a PR.

## License

[MIT](LICENSE) © 2026 Kevin Serrano
