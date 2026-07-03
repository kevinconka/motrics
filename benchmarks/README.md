# benchmarks

Validates motrics for **numeric parity** and **speed** against
[TrackEval](https://github.com/JonathonLuiten/TrackEval) and
[py-motmetrics](https://github.com/cheind/py-motmetrics), on the same sequences
the parity tests use (`fixtures.py`).

```bash
uv sync --group parity
uv run maturin develop --release --uv   # release! debug is ~10x slower
uv run python benchmarks/benchmark.py   # --repeats N, --smoke
```

Parity is a hard gate (the run fails on any mismatch); timing is reported per
sequence and as a summary. By default it uses synthetic in-memory sequences; if
real MOTChallenge sequences are present under `data/real/` it uses those:

```bash
uv run python benchmarks/download.py    # fetches TrackEval's data bundle
```

Speed figures live in the top-level [README](../README.md#benchmarks). The
comparison is deliberately conservative (TrackEval is handed pre-aligned
arrays), so real-world gains are typically larger.
