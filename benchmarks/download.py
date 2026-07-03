#!/usr/bin/env python3
"""Fetch real MOTChallenge sequences for the benchmark.

Downloads TrackEval's prepared bundle (ground truth + example tracker results,
~150 MB) and lays each sequence out as ``data/real/<seq>/{gt/gt.txt,pred.txt}``.
This is the source TrackEval documents; motchallenge.net's own zips are large,
license-gated, and unreachable from some networks. Needs network access.

    uv run python benchmarks/download.py [--split MOT17-train]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REAL_DIR = Path(__file__).parent / "data" / "real"
DATA_URL = "https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip"

_GT = re.compile(r"gt/mot_challenge/(?P<split>[^/]+)/(?P<seq>[^/]+)/gt/gt\.txt$")
_TRACKER = re.compile(
    r"trackers/mot_challenge/(?P<split>[^/]+)/[^/]+/data/(?P<seq>[^/]+)\.txt$"
)


def _download(url: str, dest: Path) -> None:
    print(f"downloading {url} ...", flush=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as handle:
        shutil.copyfileobj(resp, handle)
    print(f"  saved {dest.stat().st_size / 1e6:.1f} MB", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="MOT17-train", help="benchmark split")
    parser.add_argument("--clean", action="store_true", help="remove data/real first")
    args = parser.parse_args()

    if args.clean and REAL_DIR.exists():
        shutil.rmtree(REAL_DIR)
    elif any(REAL_DIR.glob("*/gt/gt.txt")):
        print(
            f"{REAL_DIR} already populated, skipping download (use --clean to refetch)"
        )
        return 0
    REAL_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "data.zip"
        try:
            _download(DATA_URL, zip_path)
        except OSError as exc:
            print(f"error: download failed ({exc}).", file=sys.stderr)
            print(
                f"{DATA_URL} must be reachable (blocked in the sandbox).",
                file=sys.stderr,
            )
            return 1

        with zipfile.ZipFile(zip_path) as zf:
            gt: dict[str, str] = {}
            preds: dict[str, str] = {}  # seq -> first tracker's results
            for name in zf.namelist():
                if (m := _GT.search(name)) and m["split"] == args.split:
                    gt[m["seq"]] = name
                elif (m := _TRACKER.search(name)) and m["split"] == args.split:
                    preds.setdefault(m["seq"], name)

            seqs = sorted(gt.keys() & preds.keys())
            if not seqs:
                print(
                    f"error: no gt+tracker sequences for {args.split}.", file=sys.stderr
                )
                return 1
            for seq in seqs:
                d = REAL_DIR / seq
                (d / "gt").mkdir(parents=True, exist_ok=True)
                (d / "gt" / "gt.txt").write_bytes(zf.read(gt[seq]))
                (d / "pred.txt").write_bytes(zf.read(preds[seq]))
                print(f"  {seq}")

    print(f"\n{len(seqs)} sequence(s) under {REAL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
