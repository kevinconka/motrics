"""Parity: ``motrics.compat.trackeval`` vs real TrackEval, end to end.

Builds a real MOTChallenge-style dataset directory (gt/seqinfo.ini/seqmap +
tracker results) and points both motrics' and TrackEval's own
``Evaluator``/``MotChallenge2DBox``/``metrics.{HOTA,CLEAR,Identity}`` at it —
exercising directory scanning, seqmap parsing, and preprocessing exactly as a
real evaluation script would, not just the metric math in isolation.
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any

import motrics.compat.trackeval as trackeval
import numpy as np
import pytest

real_trackeval = pytest.importorskip("trackeval")

# frame, id, left, top, w, h, conf/consider, class, visibility.
GT_MOT17_02 = [
    (1, 1, 0, 0, 10, 10, 1, 1, 1.0),  # plain pedestrian
    (1, 2, 20, 20, 10, 10, 1, 8, 1.0),  # distractor, matched by pred below
    (1, 3, 40, 40, 10, 10, 0, 1, 1.0),  # pedestrian, "do not consider"
    (2, 1, 1, 1, 10, 10, 1, 1, 1.0),
]
# frame, id, left, top, w, h, conf, x, y, z.
PRED_MOT17_02 = [
    (1, 10, 0, 0, 10, 10, 0.9, -1, -1, -1),  # matches pedestrian 1
    (1, 20, 20, 20, 10, 10, 0.9, -1, -1, -1),  # matches distractor 2 -> dropped
    (1, 30, 90, 90, 10, 10, 0.9, -1, -1, -1),  # unmatched -> real false positive
    (2, 10, 1, 1, 10, 10, 0.9, -1, -1, -1),  # continues id 10 -> no switch
]
GT_MOT17_04 = [(1, 1, 5, 5, 8, 8, 1, 1, 1.0)]
PRED_MOT17_04 = [(1, 99, 5, 5, 8, 8, 0.9, -1, -1, -1)]

SEQUENCES = {
    "MOT17-02": (GT_MOT17_02, PRED_MOT17_02),
    "MOT17-04": (GT_MOT17_04, PRED_MOT17_04),
}
TRACKER = "mytracker"


def _write_rows(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(",".join(str(v) for v in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _build_dataset_dir(tmp_path: Path) -> tuple[Path, Path]:
    gt_folder = tmp_path / "gt"
    trackers_folder = tmp_path / "trackers"
    for seq, (gt_rows, pred_rows) in SEQUENCES.items():
        _write_rows(gt_folder / "MOT17-train" / seq / "gt" / "gt.txt", gt_rows)
        _write_rows(
            trackers_folder / "MOT17-train" / TRACKER / "data" / f"{seq}.txt", pred_rows
        )
        num_timesteps = max(row[0] for row in gt_rows)
        ini = configparser.ConfigParser()
        ini["Sequence"] = {"seqLength": str(num_timesteps)}
        ini_path = gt_folder / "MOT17-train" / seq / "seqinfo.ini"
        with open(ini_path, "w", encoding="utf-8") as f:
            ini.write(f)
    seqmap = gt_folder / "seqmaps" / "MOT17-train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(SEQUENCES) + "\n", encoding="utf-8")
    return gt_folder, trackers_folder


def _dataset_config(gt_folder: Path, trackers_folder: Path) -> dict[str, Any]:
    return {
        "GT_FOLDER": str(gt_folder),
        "TRACKERS_FOLDER": str(trackers_folder),
        "PRINT_CONFIG": False,
    }


def _run(module: Any, gt_folder: Path, trackers_folder: Path) -> dict[str, Any]:
    evaluator = module.Evaluator(
        {**module.Evaluator.get_default_eval_config(), "PRINT_CONFIG": False}
    )
    dataset = module.datasets.MotChallenge2DBox(
        _dataset_config(gt_folder, trackers_folder)
    )
    metrics_list = [
        module.metrics.HOTA(),
        module.metrics.CLEAR(),
        module.metrics.Identity(),
    ]
    results, messages = evaluator.evaluate([dataset], metrics_list)
    assert messages["MotChallenge2DBox"][TRACKER] == "Success"
    return results["MotChallenge2DBox"][TRACKER]


def test_matches_real_trackeval_end_to_end(tmp_path: Path) -> None:
    gt_folder, trackers_folder = _build_dataset_dir(tmp_path)
    mine = _run(trackeval, gt_folder, trackers_folder)
    real = _run(real_trackeval, gt_folder, trackers_folder)

    for seq_key in [*SEQUENCES, "COMBINED_SEQ"]:
        m = mine[seq_key]["pedestrian"]
        r = real[seq_key]["pedestrian"]
        for metric_name, fields in [
            ("CLEAR", ["MOTA", "MOTP", "CLR_TP", "CLR_FN", "CLR_FP", "IDSW"]),
            ("Identity", ["IDF1", "IDP", "IDR", "IDTP", "IDFP", "IDFN"]),
            (
                "HOTA",
                [
                    "HOTA",
                    "DetA",
                    "AssA",
                    "LocA",
                    "AssRe",
                    "AssPr",
                    "HOTA_TP",
                    "HOTA_FN",
                    "HOTA_FP",
                ],
            ),
        ]:
            for field in fields:
                mine_v, real_v = m[metric_name][field], r[metric_name][field]
                assert np.allclose(np.asarray(mine_v), np.asarray(real_v), atol=1e-9), (
                    f"{seq_key}/{metric_name}/{field}: {mine_v} != {real_v}"
                )

    # Sanity check the scenario actually exercises distractor/consider filtering
    # and a continuing track (no id switch) rather than being trivially empty.
    assert mine["MOT17-02"]["pedestrian"]["CLEAR"]["CLR_TP"] == 2
    assert mine["MOT17-02"]["pedestrian"]["CLEAR"]["CLR_FP"] == 1
    assert mine["MOT17-02"]["pedestrian"]["CLEAR"]["IDSW"] == 0


def test_do_preproc_false_raises(tmp_path: Path) -> None:
    gt_folder, trackers_folder = _build_dataset_dir(tmp_path)
    config = {**_dataset_config(gt_folder, trackers_folder), "DO_PREPROC": False}
    with pytest.raises(trackeval.TrackEvalException):
        trackeval.datasets.MotChallenge2DBox(config)


def test_mot15_benchmark_raises(tmp_path: Path) -> None:
    gt_folder, trackers_folder = _build_dataset_dir(tmp_path)
    config = {**_dataset_config(gt_folder, trackers_folder), "BENCHMARK": "MOT15"}
    with pytest.raises(trackeval.TrackEvalException):
        trackeval.datasets.MotChallenge2DBox(config)
