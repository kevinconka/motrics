"""Ensure the shared benchmark fixtures exist before the tests that load them.

The synthetic MOTChallenge fixtures under ``benchmarks/data`` are generated (not
committed), so regenerate them on demand for the parity tests. Runs the
generator as a subprocess at collection time, before ``test_parity`` imports and
discovers them.
"""

import subprocess
import sys
from pathlib import Path

_BENCHMARKS = Path(__file__).parents[1] / "benchmarks"
_DATA = _BENCHMARKS / "data"


def _has_fixtures() -> bool:
    if not _DATA.is_dir():
        return False
    return any(
        (d / "gt" / "gt.txt").is_file() and (d / "pred.txt").is_file()
        for d in _DATA.iterdir()
        if d.is_dir() and d.name != "real"
    )


if _BENCHMARKS.is_dir() and not _has_fixtures():
    subprocess.run(
        [sys.executable, str(_BENCHMARKS / "generate_fixtures.py")], check=True
    )
