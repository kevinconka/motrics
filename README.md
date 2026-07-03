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
`iou`, `iou_matrix`, and `match_boxes` are also exposed. Need a precomputed
similarity matrix instead of boxes (e.g. a custom Re-ID distance)?
`compute_clear_from_similarity` and `compute_identity_from_similarity` take
that directly.

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

`pip install motrics[compat]` (pulls in pandas, needed only for this
subpackage). Supported metric names: `mota`, `motp`, `idf1`, `idp`, `idr`,
`recall`, `precision`, `num_false_positives`, `num_misses`, `num_switches`,
`num_unique_objects` — the same names py-motmetrics uses. Per-trajectory
metrics (`mostly_tracked`, `partially_tracked`, `mostly_lost`,
`num_fragmentations`, `num_transfer`, `num_ascend`, `num_migrate`) aren't
implemented yet; requesting them raises `NotImplementedError` naming exactly
what's missing, rather than a silently wrong number. See
[`python/motrics/compat/motmetrics/`](python/motrics/compat/motmetrics/) for
what else differs (e.g. no `events`/`mot_events` DataFrame).

### Reading numbers from TrackEval or motmetrics against motrics' native API

If you're using `motrics`' own API directly (faster than the compat layer —
no per-frame Python bookkeeping) rather than `compat.motmetrics`, here's how
the field names line up:

| Concept                      | TrackEval            | py-motmetrics         | `motrics` (native)                        |
| ----------------------------- | --------------------- | ---------------------- | ------------------------------------------ |
| Matched detections (incl. switches) | `CLR_TP`         | `num_detections`       | `ClearMetrics.num_matches`                 |
| False positives               | `CLR_FP`               | `num_false_positives`  | `ClearMetrics.num_false_positives`         |
| Misses                         | `CLR_FN`               | `num_misses`           | `ClearMetrics.num_misses`                  |
| Identity switches              | `IDSW`                 | `num_switches`         | `ClearMetrics.num_switches`                |
| MOTA / MOTP                    | `MOTA` / `MOTP`        | `mota` / `motp`        | `ClearMetrics.mota` / `.motp`              |
| Identity TP / FP / FN          | `IDTP`/`IDFP`/`IDFN`   | `idtp`/`idfp`/`idfn`   | `IdentityMetrics.idtp`/`.idfp`/`.idfn`     |
| IDF1 / IDP / IDR               | `IDF1`/`IDP`/`IDR`     | `idf1`/`idp`/`idr`     | `IdentityMetrics.idf1`/`.idp`/`.idr`       |
| HOTA / DetA / AssA / LocA      | `HOTA`/`DetA`/`AssA`/`LocA` | — (not in motmetrics) | `HotaMetrics.hota`/`.deta`/`.assa`/`.loca` |

## Benchmarks

Roughly how much faster motrics is than TrackEval and py-motmetrics on real
MOT17 data (release build; illustrative, machine-dependent):

| motrics vs…   | CLEAR + Identity | With HOTA |
| ------------- | ---------------- | --------- |
| TrackEval     | ~3–4×            | ~6×       |
| py-motmetrics | ~14×             | —         |

See [`benchmarks/README.md`](benchmarks/README.md) for methodology, numbers,
and how to run it yourself.

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
  - [x] Precomputed-similarity core inputs (`compute_clear_from_similarity`,
        `compute_identity_from_similarity`) — the piece `compat.motmetrics`
        needed, and the first slice of "broaden core inputs" below.
  - [x] `motrics.compat.motmetrics` — a drop-in `MOTAccumulator` replacement.
        See "Migrating from py-motmetrics" above.
  - [x] Migration guide + metric-name map (see above).
  - [ ] `motrics.compat.trackeval` — a drop-in for the MOTChallenge evaluation
        path (`Evaluator`/dataset/metrics), no TrackEval installed. Gated on
        TrackEval-parity MOTChallenge preprocessing below.
  - [ ] Broaden core inputs further (`xywh` boxes, zero-copy NumPy) so users
        pass what they already hold.
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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, tooling, and
checks to run before opening a PR.

## License

[MIT](LICENSE) © 2026 Kevin Serrano
