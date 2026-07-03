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
top-level [README](../README.md#benchmarks) — the comparison is conservative
(TrackEval is handed pre-aligned arrays), so real-world gains are typically
larger.

Exact numeric parity is also checked on synthetic sequences in
`tests/test_parity.py` (`fixtures.make_synthetic`), which run offline in CI.

## Notes

- **Tie-breaking, not a bug.** On dense real sequences, Hungarian assignment
  ties are resolved differently by each solver (`lsap` vs `scipy`), shifting a
  couple of matches/switches out of thousands (e.g. IDSW 70 vs 72 with
  identical TP/FP/FN). `benchmark.py` tolerates this (`PARITY_ATOL`);
  `tests/test_parity.py` still enforces exact 1e-9 parity on tie-free synthetic
  data, which is the real bug-catching gate.
- **Measured speedups (MOT17-train, release build):** motrics runs roughly
  3–9× faster than TrackEval and 13–30× faster than py-motmetrics. Exact ratios
  vary by sequence density and machine — rerun the benchmark for current numbers
  rather than trusting these as fixed.
