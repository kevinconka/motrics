<h1 align="center">motrics</h1>

<p align="center">
  <em>An extremely fast MOT and HOTA metrics library, written in Rust:
  CLEAR (MOTA/MOTP), Identity (IDF1), and HOTA, with an ergonomic Python
  API.</em>
</p>

<p align="center">
  <a href="https://github.com/kevinconka/motrics/actions/workflows/ci.yml"><img src="https://github.com/kevinconka/motrics/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/kevinconka/motrics"><img src="https://codecov.io/gh/kevinconka/motrics/graph/badge.svg" alt="codecov"></a>
  <a href="https://pypi.org/project/motrics/"><img src="https://img.shields.io/pypi/v/motrics" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/kevinconka/motrics" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="benchmarks/assets/speedup-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="benchmarks/assets/speedup-light.svg">
    <img alt="Bar chart: motrics computes CLEAR+Identity+HOTA in 770ms vs TrackEval's 5930ms (7.7x faster), and CLEAR+Identity in 443ms vs py-motmetrics' 6211ms (14.0x faster)." src="benchmarks/assets/speedup-light.svg" width="480">
  </picture>
</p>
<p align="center"><i>MOT17-train, wall time, from a live CI run. See <a href="#benchmarks">Benchmarks</a>.</i></p>

## Highlights

- **Fast.** Rust core; see [Benchmarks](#benchmarks) for numbers against
  TrackEval and py-motmetrics.
- **Validated.** Bit-exact parity with TrackEval on CLEAR, Identity, and HOTA,
  checked in CI.
- **Drop-in migration.** Swap one import to replace py-motmetrics or
  TrackEval.
- **Typed Python API.** PEP 561, `numpy` the only required runtime
  dependency.
- **Flexible box input.** `xyxy` or `xywh`, and a zero-copy read path for
  contiguous NumPy arrays.

## Install

```bash
pip install motrics
```

Prebuilt wheels for Linux, macOS, and Windows (Python 3.10+). Building from
source instead? See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup.

## Quickstart

```python
import motrics

# Parse MOTChallenge ground truth and tracker results.
gt = motrics.load_motchallenge("seq/gt/gt.txt")
pred = motrics.load_motchallenge("seq/res.txt", min_confidence=0.5)

# Align onto a shared frame timeline, bundle each side, then evaluate.
gt_ids, gt_boxes, pred_ids, pred_boxes = motrics.align_frames(gt, pred)
result = motrics.evaluate(
    motrics.Frames(ids=gt_ids, boxes=gt_boxes),
    motrics.Frames(ids=pred_ids, boxes=pred_boxes),
)

print(result.clear.mota, result.identity.idf1, result.hota.hota)
```

- Only need one metric? `compute_clear`/`compute_identity`/`compute_hota` take
  the same four arguments directly, no `Frames` needed.
- Boxes: `xyxy` by default, `box_format="xywh"` for the alternative; NumPy
  `(N, 4)` arrays accepted too.
- Want TrackEval's exact reported numbers? Use `load_motchallenge_gt` +
  `preprocess_motchallenge` instead of `load_motchallenge` + `align_frames`.

## Datasets

Each dataset gets a small adapter (ingest + preprocessing + similarity) on
top of the shared metric core. All are validated against TrackEval's own
preprocessing:

| Dataset | Box/mask | Similarity | Load                                     |
| ------- | -------- | ---------- | ----------------------------------------- |
| MOTChallenge / DanceTrack | box | IoU | `load_motchallenge(_gt)` |
| KITTI 2D-box | box | IoU | `load_kitti(_gt)` |
| KITTI-MOTS | mask (RLE) | mask IoU | `load_kitti_mots(_gt)` |
| DAVIS (unsupervised) | indexed PNG mask | mask IoU | `load_davis` |
| BDD100K | box (JSON) | IoU | `load_bdd100k(_gt)` |
| KITTI-3D | oriented 3D box | volumetric IoU | `load_kitti_3d(_gt)` |

Each adapter pairs with a matching `preprocess_*` function that applies the
dataset's TrackEval preprocessing rules (distractor classes, occlusion/
truncation thresholds, ignore regions, ...) and returns arguments for
`compute_clear`/`compute_identity`/`compute_hota` or their
`_from_similarity` counterparts.

## Migrating from py-motmetrics or TrackEval

Swap one import, the rest of your code is unchanged.

<details>
<summary>py-motmetrics</summary>

```python
# before
import motmetrics as mm

# after: same code, motrics underneath
import motrics.compat.motmetrics as mm

acc = mm.MOTAccumulator(auto_id=True)
for gt_ids, gt_boxes, pred_ids, pred_boxes in sequence:
    dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
    acc.update(gt_ids, pred_ids, dists)

summary = mm.metrics.create().compute(acc, metrics=mm.metrics.SUPPORTED, name="acc")
```

`pip install motrics[compat]` (pulls in pandas, needed only for this subpackage).

The full `motmetrics.metrics.motchallenge_metrics` field set is supported:
`mota`, `motp`, `idf1`, `idp`, `idr`, `recall`, `precision`,
`num_false_positives`, `num_misses`, `num_switches`, `num_unique_objects`,
`mostly_tracked`, `partially_tracked`, `mostly_lost`, `num_fragmentations`,
`num_transfer`, `num_ascend`, `num_migrate`.

See [`python/motrics/compat/motmetrics/`](python/motrics/compat/motmetrics/)
for what else differs (e.g. no `events`/`mot_events` DataFrame).

</details>

<details>
<summary>TrackEval</summary>

```python
# before
import trackeval

# after: same code, motrics underneath
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

Same class names, config keys, directory/seqmap conventions, and result shape
as real TrackEval. No `trackeval`/`scipy` install required, only `numpy` (a
core dependency already).

| | |
| --- | --- |
| ✅ Supported | `HOTA`, `Identity`, and `CLEAR`'s full field set (`MOTA`/`MOTP`/`MODA`/`sMOTA`/`MOTAL`, `MT`/`PT`/`ML`/`Frag`, `CLR_Re`/`CLR_Pr`/`MTR`/`PTR`/`MLR`/`CLR_F1`/`FP_per_frame`), bit-exact vs real TrackEval |
| ❌ Not yet | Parallel evaluation, `BREAK_ON_ERROR` config, printing/plotting, zipped input, `DO_PREPROC=False`, `MOT15`, `IDEucl`/`JAndF`/`TrackMAP`/`VACE` |

See [`python/motrics/compat/trackeval/`](python/motrics/compat/trackeval/)
for the full list of what differs from real TrackEval.

</details>

<details>
<summary>Metric name map: TrackEval / py-motmetrics / motrics' native API</summary>

Using `motrics`' own API directly (faster than the compat layer, no
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
| HOTA / DetA / AssA / LocA             | `HOTA`/`DetA`/`AssA`/`LocA` | (not in motmetrics)    | `HotaMetrics.hota`/`.deta`/`.assa`/`.loca` |

</details>

## Benchmarks

On real MOT17 data, release build, end-to-end from raw boxes (chart at the top
of this README):

| motrics vs…   | Metrics                | Speedup |
| ------------- | ----------------------- | ------- |
| TrackEval     | CLEAR + Identity + HOTA | ~7–9×   |
| py-motmetrics | CLEAR + Identity        | ~12–16× |

Numbers are illustrative and machine-dependent. See the CI benchmark comment
on any PR for a live measurement, and
[`benchmarks/README.md`](benchmarks/README.md) for methodology and how to run
it yourself.

<details>
<summary>Roadmap</summary>

- [x] Project scaffolding (build, lint, packaging, CI)
- [x] Published to PyPI, automated tag-and-release on every `Cargo.toml`
      version bump
- [x] Box IoU + assignment (Hungarian/greedy) primitives
- [x] CLEAR metrics: MOTA/MOTP, ID switches, FP/FN, MT/PT/ML, fragmentations,
      and the derived MODA/sMOTA/MOTAL/CLR_Re/CLR_Pr fields
- [x] Identity metrics (IDF1/IDP/IDR)
- [x] HOTA (DetA, AssA, alpha sweep)
- [x] MOTChallenge ingest, integration tests, TrackEval numeric parity tests
- [x] Benchmark & parity infrastructure vs TrackEval and py-motmetrics on real
      MOTChallenge data, validated in CI
- [x] Zero-copy NumPy input path; `xyxy`/`xywh` box formats
- [x] Precomputed-similarity core inputs (`compute_*_from_similarity`)
- [x] `motrics.compat.motmetrics`: drop-in `MOTAccumulator`, full
      `motchallenge_metrics` field set including switch subtypes
      (`num_transfer`/`num_ascend`/`num_migrate`)
- [x] `motrics.compat.trackeval`: drop-in `Evaluator`/`MotChallenge2DBox`/
      `HOTA`/`CLEAR`/`Identity`
- [x] Ergonomic native API: `Frames`, `evaluate()`, streaming `Accumulator`
      for CLEAR + Identity
- [x] Mask-IoU similarity kernel (RLE codec, ignore-region/IoA semantics) for
      mask-based adapters
- [x] 3D IoU similarity kernel (oriented boxes) for KITTI-3D
- [x] Dataset adapters: DanceTrack, KITTI 2D-box, KITTI-MOTS, DAVIS, BDD100K,
      KITTI-3D (see [Datasets](#datasets))
- [ ] Other TrackEval metrics: `IDEucl`, `TrackMAP`, `VACE`, `JAndF` (DAVIS's
      native metric)
- [ ] Remaining `compat.trackeval` `Evaluator` behaviors: parallel
      evaluation, printing/plotting, zipped input, `DO_PREPROC=False`,
      `MOT15`, `BREAK_ON_ERROR`

</details>

## Acknowledgments

motrics reimplements the evaluation protocols and metric definitions of two
projects. All credit for the underlying methodology belongs to their
authors; motrics is a from-scratch Rust port, not a wrapper around either.

- [TrackEval](https://github.com/JonathonLuiten/TrackEval): the reference
  implementation for CLEAR, Identity, and HOTA, and for every dataset
  adapter's preprocessing rules. motrics is validated against it in CI.
- [py-motmetrics](https://github.com/cheind/py-motmetrics): the reference
  implementation for the MOTChallenge metric set and the `MOTAccumulator`
  API that `compat.motmetrics` mirrors.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, tooling,
and checks to run before opening a PR.

## License

[MIT](LICENSE) © 2026 Kevin Serrano
