# benchmarks

A reproducible suite that validates `motrics` for **numeric parity** against
reference MOT evaluators and measures its **speed** on MOTChallenge-format data.

Everything here shares one set of inputs with the parity tests: the synthetic
fixtures under `data/`, loaded through the public `motrics.load_motchallenge`
reader — the same reader used for real MOTChallenge sequences. So parity and
benchmarking run on identical, MOTChallenge-format inputs rather than two
separate synthetic generators.

## Layout

```text
benchmarks/
  fixtures.py            # dataset discovery + loading (shared)
  generate_fixtures.py   # (re)generate the synthetic fixtures
  benchmark.py           # parity + timing runner
  download.py            # fetch real MOTChallenge sequences (needs network)
  data/                  # generated, git-ignored (not committed)
    <sequence>/gt/gt.txt
    <sequence>/pred.txt
    real/                # real sequences land here (opt-in)
```

A *dataset* is a directory of sequence folders, each with `gt/gt.txt` and
`pred.txt`. `benchmark.py` uses `data/real/` when it is present and falls back
to the synthetic fixtures otherwise. The synthetic fixtures are **git-ignored
and generated on demand** — `benchmark.py` (and the parity tests) create them
automatically, or run `generate_fixtures.py` yourself.

## Running

```bash
uv sync --group parity                      # numpy, scipy, trackeval, motmetrics
uv run maturin develop --release --uv       # RELEASE build — see note below
uv run python benchmarks/benchmark.py       # auto: real data if present, else fixtures
```

> **Build in release mode.** A debug build (`maturin develop` without
> `--release`) is roughly 10× slower, which makes the timings meaningless. The
> benchmark calls `motrics.is_debug_build()` and prints a loud warning if it
> detects one.

Useful flags:

- `--smoke` — one repeat, just verifies the suite runs (used in CI).
- `--repeats N` — timing repeats per engine (best time is reported).
- `--data PATH` — point at an explicit dataset root.
- `--min-confidence C` — drop tracker detections below confidence `C`.

TrackEval and py-motmetrics are optional; each is skipped with a note if absent.

## Real MOTChallenge data

Real datasets are large and under a **research, non-redistribution license**
(CC BY-NC-SA), so they are **not** committed here. Fetch them explicitly (needs
network access to `motchallenge.net`, which is **blocked in the default sandbox**
— run in CI or a permissioned local session):

```bash
uv run python benchmarks/download.py --dataset MOT15          # smallest
uv run python benchmarks/download.py --dataset MOT17 --sequences MOT17-04 MOT17-09
```

By default `download.py` seeds `pred.txt` from ground truth (a perfect tracker,
useful for exercising the pipeline). Pass `--tracker-zip URL` to benchmark
against real tracker output instead.

## What is measured

For each sequence, every available engine computes three metric families and is
timed end-to-end (from frame-aligned detections to metrics, **including each
engine's own similarity computation**):

| Engine          | CLEAR | Identity | HOTA | Similarity |
| --------------- | :---: | :------: | :--: | ---------- |
| motrics         |   ✓   |    ✓     |  ✓   | Rust, internal |
| TrackEval       |   ✓   |    ✓     |  ✓   | vectorised NumPy |
| py-motmetrics   |   ✓   |    ✓     |  —   | vectorised NumPy |

**Parity** is a hard gate (the runner exits non-zero on mismatch). motrics vs
TrackEval is checked tightly — TrackEval is fed *identical* similarities from
`motrics.iou_matrix`, so any gap would be pure metric-math. py-motmetrics uses
its own matching and a distance-based MOTP, so it is checked with a looser
tolerance and MOTP is excluded.

**Timing** lets each engine use its natural similarity path so the comparison
is apples-to-apples. Two speedup views are reported: `CLEAR+Identity` (all three
engines) and the full pipeline including HOTA (motrics vs TrackEval).

### Example output (release build, synthetic fixtures — numbers are machine-dependent)

```text
MOTRICS-SYNTH-03  (500 frames, 9215 gt / 7904 pred dets)
    metrics: MOTA=0.7703 MOTP=0.7532 IDF1=0.8749 HOTA=0.5949
    parity: OK (all engines agree within tolerance)
    engine        all families    CLEAR+Id  speedup*
    motrics            15.59ms       8.69ms      1.0x
    trackeval         178.87ms      49.13ms      5.7x
    motmetrics        205.99ms     205.99ms     23.7x  no HOTA

SUMMARY (speedup = engine time / motrics time; higher = motrics faster)
  motrics    1.0x   ·   trackeval 6.3x   ·   motmetrics 27.5x   (CLEAR+Identity)
  Full pipeline incl. HOTA (motrics vs TrackEval): 13.0x faster
```

### Reading the numbers

On these fixtures motrics is roughly **6× faster than TrackEval** on the
CLEAR+Identity subset (**~13×** on the full pipeline including HOTA) and
**~25× faster than the classic py-motmetrics**. Exact ratios vary with sequence
size and machine.

This is deliberately a **conservative** comparison for motrics: it isolates
metric computation and hands TrackEval pre-aligned, contiguous arrays.
TrackEval's full end-to-end path (file parsing, preprocessing, its own
similarity stage over all detections) carries overhead this harness excludes, so
real-world end-to-end gains are typically larger.

> **These numbers require a release build.** A debug build makes motrics ~10×
> slower and can reverse the comparison; the benchmark warns if it detects one.

Accepting NumPy arrays with a zero-copy input path (and sharing one similarity
matrix across the metric families, instead of the current per-call Python-list
marshalling) is the identified next optimization — see the roadmap in the
top-level `README.md`.
