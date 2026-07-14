# benchmarks

Benchmarks motrics for **speed** and **numeric parity** against
[TrackEval](https://github.com/JonathonLuiten/TrackEval) and
[py-motmetrics](https://github.com/cheind/py-motmetrics) on real MOTChallenge
sequences.

```bash
uv sync --group parity
uv run maturin develop --release --uv    # release! debug is ~10x slower
uv run python benchmarks/download.py     # fetch MOT17-train (TrackEval's bundle)
uv run python benchmarks/benchmark.py    # --repeats N, --smoke
```

Parity is a hard gate (fails on any mismatch beyond assignment tie-breaking);
timing is reported per sequence and as a summary. Speed figures live in the
top-level [README](../README.md#benchmarks). The comparison is conservative
(TrackEval is handed pre-aligned arrays), so real-world gains are typically
larger.

Exact numeric parity is also checked on synthetic sequences in
`tests/test_parity.py` (`fixtures.make_synthetic`), which run offline in CI.

## CI

Every pull request runs this benchmark (real data cached across runs) and
posts the results as a sticky comment, updated in place on each push rather
than a new comment every time. See the `benchmark` job in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml). It's informational,
not a merge gate: CI runner timing has real variance, so treat the numbers as
a rough signal rather than a precise regression check. `--markdown PATH`
writes the same report `benchmark.py` prints to stdout as a markdown file for
that comment.

## Chart

The bar chart in the top-level README (`benchmarks/assets/speedup-*.svg`) is
generated from `plot_speedup.py`, not measured at doc-build time. After a
benchmark run shows the numbers have meaningfully drifted, update `DATA` in
that script and regenerate:

```bash
uv sync --group plot
uv run python benchmarks/plot_speedup.py
```

## Notes

- **Ground truth is preprocessed.** `fixtures.load_real()` runs gt.txt through
  `preprocess_motchallenge` (distractor-class removal, pedestrian-only, "do
  not consider" rows dropped) before scoring, matching what TrackEval's own
  `MotChallenge2DBox` evaluates rather than raw box IoU.
- **Tie-breaking, not a bug.** On dense real sequences, Hungarian assignment
  ties are resolved differently by each solver (`lsap` vs `scipy`), shifting a
  couple of matches/switches out of thousands (e.g. IDSW 70 vs 72 with
  identical TP/FP/FN). `benchmark.py` tolerates this (`PARITY_ATOL`);
  `tests/test_parity.py` still enforces exact 1e-9 parity on tie-free synthetic
  data, which is the real bug-catching gate.
- **Measured speedups (MOT17-train, release build):** motrics runs roughly
  3–9× faster than TrackEval and 13–30× faster than py-motmetrics. Exact
  ratios vary by sequence density and machine; rerun the benchmark for
  current numbers rather than trusting these as fixed.
