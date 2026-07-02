#!/usr/bin/env python3
"""Fetch real MOTChallenge sequences for the benchmark and parity suite.

MOTChallenge datasets are large and distributed under a **research,
non-redistribution license** (CC BY-NC-SA), so they are *not* committed to this
repository. This script downloads a chosen dataset zip from
``https://motchallenge.net`` and arranges each sequence into the layout the
benchmark expects::

    benchmarks/data/real/<sequence>/gt/gt.txt     # ground truth
    benchmarks/data/real/<sequence>/pred.txt      # tracker results

Ground truth comes straight from the dataset's ``gt/gt.txt``. A tracker results
file is needed for the ``pred.txt`` side; by default this script seeds ``pred``
from ``gt`` so the pipeline is exercised end-to-end (a perfect tracker). Pass
``--tracker-zip URL`` to instead unpack real tracker output (e.g. a MOTChallenge
results submission), matching sequences by name.

Network access to ``motchallenge.net`` is required and must be permitted by the
environment's egress policy — it is blocked in the default sandbox, so run this
in CI or a permissioned local session. Nothing here is imported by the tests;
they fall back to the generated synthetic fixtures when ``real/`` is absent.

Examples::

    uv run python benchmarks/download.py --dataset MOT15
    uv run python benchmarks/download.py --dataset MOT17 --sequences MOT17-04 MOT17-09
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REAL_DIR = Path(__file__).parent / "data" / "real"
BASE_URL = "https://motchallenge.net/data"

# Known dataset zips (train split carries ground truth). MOT15's train dir is
# named ``2DMOT2015``; the others follow the ``<name>/train`` convention.
DATASETS = {
    "MOT15": "2DMOT2015.zip",
    "MOT16": "MOT16.zip",
    "MOT17": "MOT17.zip",
    "MOT20": "MOT20.zip",
}


def _download(url: str) -> bytes:
    print(f"downloading {url} ...", flush=True)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    print(f"  got {len(data) / 1e6:.1f} MB", flush=True)
    return data


def _find_gt_files(zf: zipfile.ZipFile) -> dict[str, str]:
    """Map sequence name -> member path of its ``gt/gt.txt`` inside the zip."""
    found: dict[str, str] = {}
    for name in zf.namelist():
        parts = name.split("/")
        # .../<sequence>/gt/gt.txt
        if len(parts) >= 3 and parts[-2:] == ["gt", "gt.txt"]:
            found[parts[-3]] = name
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="MOT15",
        help="MOTChallenge dataset to fetch (default: MOT15, the smallest)",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="only extract these sequences (default: all with ground truth)",
    )
    parser.add_argument(
        "--tracker-zip",
        default=None,
        help="URL of a tracker-results zip; sequences matched by name for pred.txt",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove benchmarks/data/real before extracting",
    )
    args = parser.parse_args()

    if args.clean and REAL_DIR.exists():
        shutil.rmtree(REAL_DIR)
    REAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        gt_zip_bytes = _download(f"{BASE_URL}/{DATASETS[args.dataset]}")
    except OSError as exc:  # network blocked / host unreachable
        print(f"error: download failed ({exc}).", file=sys.stderr)
        print(
            "motchallenge.net must be reachable and allowed by the egress policy; "
            "it is blocked in the default sandbox. Run this in CI or a "
            "permissioned session.",
            file=sys.stderr,
        )
        return 1

    tracker_zf: zipfile.ZipFile | None = None
    if args.tracker_zip:
        tracker_zf = zipfile.ZipFile(io.BytesIO(_download(args.tracker_zip)))

    count = 0
    with zipfile.ZipFile(io.BytesIO(gt_zip_bytes)) as zf:
        gt_files = _find_gt_files(zf)
        if not gt_files:
            print("error: no gt/gt.txt found in the zip.", file=sys.stderr)
            return 1
        wanted = set(args.sequences) if args.sequences else set(gt_files)
        for seq, member in sorted(gt_files.items()):
            if seq not in wanted:
                continue
            seq_dir = REAL_DIR / seq
            (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
            (seq_dir / "gt" / "gt.txt").write_bytes(zf.read(member))

            # pred.txt: from a tracker zip if given, else seed from gt.
            pred_written = False
            if tracker_zf is not None:
                for tname in tracker_zf.namelist():
                    if Path(tname).stem == seq and tname.endswith(".txt"):
                        (seq_dir / "pred.txt").write_bytes(tracker_zf.read(tname))
                        pred_written = True
                        break
            if not pred_written:
                shutil.copyfile(seq_dir / "gt" / "gt.txt", seq_dir / "pred.txt")
            count += 1
            print(f"  extracted {seq}")

    if tracker_zf is not None:
        tracker_zf.close()

    print(f"\ndone: {count} sequence(s) under {REAL_DIR}")
    print("run: uv run python benchmarks/benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
