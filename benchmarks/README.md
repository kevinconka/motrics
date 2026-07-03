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
