"""``MotChallenge2DBox`` dataset, mirroring
``trackeval.datasets.mot_challenge_2d_box.MotChallenge2DBox``.

Directory scanning, seqmap parsing, and ``seqinfo.ini`` reading follow
TrackEval's own conventions exactly (``GT_FOLDER/BENCHMARK-SPLIT/<seq>/gt/gt.txt``,
a seqmap CSV listing sequence names, per-sequence ``seqinfo.ini`` for frame
counts). Preprocessing delegates to :func:`motrics.load_motchallenge_gt` /
:func:`motrics.load_motchallenge` / :func:`motrics.preprocess_motchallenge`.

Not implemented: ``INPUT_AS_ZIP`` (zipped tracker/gt input) and classes other
than ``pedestrian`` — TrackEval's own MOT Challenge adapter only supports
pedestrian too, so this isn't a gap versus TrackEval itself.
"""

from __future__ import annotations

import configparser
import csv
import os
from typing import Any

from motrics import load_motchallenge, load_motchallenge_gt, preprocess_motchallenge
from motrics.compat.trackeval._utils import TrackEvalException, init_config


class MotChallenge2DBox:
    @staticmethod
    def get_default_dataset_config() -> dict[str, Any]:
        code_path = os.getcwd()
        return {
            "GT_FOLDER": os.path.join(code_path, "data/gt/mot_challenge/"),
            "TRACKERS_FOLDER": os.path.join(code_path, "data/trackers/mot_challenge/"),
            "OUTPUT_FOLDER": None,
            "TRACKERS_TO_EVAL": None,
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "PRINT_CONFIG": True,
            "DO_PREPROC": True,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "TRACKER_DISPLAY_NAMES": None,
            "SEQMAP_FOLDER": None,
            "SEQMAP_FILE": None,
            "SEQ_INFO": None,
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
            "SKIP_SPLIT_FOL": False,
        }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = init_config(
            config, self.get_default_dataset_config(), self.get_name()
        )

        self.benchmark = self.config["BENCHMARK"]
        self.gt_set = f"{self.benchmark}-{self.config['SPLIT_TO_EVAL']}"
        split_fol = "" if self.config["SKIP_SPLIT_FOL"] else self.gt_set
        self.gt_fol = os.path.join(self.config["GT_FOLDER"], split_fol)
        self.tracker_fol = os.path.join(self.config["TRACKERS_FOLDER"], split_fol)
        self.should_classes_combine = False
        self.use_super_categories = False
        self.do_preproc = self.config["DO_PREPROC"]

        self.output_fol = self.config["OUTPUT_FOLDER"] or self.tracker_fol
        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]

        valid_classes = ["pedestrian"]
        self.class_list = [c.lower() for c in self.config["CLASSES_TO_EVAL"]]
        if not all(c in valid_classes for c in self.class_list):
            raise TrackEvalException(
                "Attempted to evaluate an invalid class. Only pedestrian is valid."
            )

        self.seq_list, self.seq_lengths = self._get_seq_info()
        if not self.seq_list:
            raise TrackEvalException("No sequences are selected to be evaluated.")
        for seq in self.seq_list:
            gt_file = self.config["GT_LOC_FORMAT"].format(
                gt_folder=self.gt_fol, seq=seq
            )
            if not os.path.isfile(gt_file):
                raise TrackEvalException(f"GT file not found for sequence: {seq}")

        if self.config["TRACKERS_TO_EVAL"] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config["TRACKERS_TO_EVAL"]

        display_names = self.config["TRACKER_DISPLAY_NAMES"]
        if display_names is None:
            self.tracker_to_disp = dict(
                zip(self.tracker_list, self.tracker_list, strict=True)
            )
        elif self.config["TRACKERS_TO_EVAL"] is not None and len(display_names) == len(
            self.tracker_list
        ):
            self.tracker_to_disp = dict(
                zip(self.tracker_list, display_names, strict=True)
            )
        else:
            raise TrackEvalException(
                "List of tracker files and tracker display names do not match."
            )

        for tracker in self.tracker_list:
            for seq in self.seq_list:
                pred_file = os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol, f"{seq}.txt"
                )
                if not os.path.isfile(pred_file):
                    raise TrackEvalException(
                        f"Tracker file not found: {tracker}/{seq}.txt"
                    )

    def get_name(self) -> str:
        return type(self).__name__

    def get_display_name(self, tracker: str) -> str:
        return self.tracker_to_disp[tracker]

    def get_output_fol(self, tracker: str) -> str:
        return str(os.path.join(self.output_fol, tracker, self.output_sub_fol))

    def get_eval_info(self) -> tuple[list[str], list[str], list[str]]:
        return self.tracker_list, self.seq_list, self.class_list

    def _get_seq_info(self) -> tuple[list[str], dict[str, int]]:
        seq_list: list[str] = []
        seq_lengths: dict[str, int] = {}
        seq_info = self.config["SEQ_INFO"]
        if seq_info:
            seq_list = list(seq_info.keys())
            seq_lengths = dict(seq_info)
            for seq, length in seq_lengths.items():
                if length is None:
                    seq_lengths[seq] = self._read_seqinfo_length(seq)
            return seq_list, seq_lengths

        seqmap_file = self.config["SEQMAP_FILE"]
        if not seqmap_file:
            seqmap_folder = self.config["SEQMAP_FOLDER"] or os.path.join(
                self.config["GT_FOLDER"], "seqmaps"
            )
            seqmap_file = os.path.join(seqmap_folder, f"{self.gt_set}.txt")
        if not os.path.isfile(seqmap_file):
            raise TrackEvalException(f"no seqmap found: {seqmap_file}")
        with open(seqmap_file, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.reader(f)):
                if i == 0 or not row or row[0] == "":
                    continue
                seq = row[0]
                seq_list.append(seq)
                seq_lengths[seq] = self._read_seqinfo_length(seq)
        return seq_list, seq_lengths

    def _read_seqinfo_length(self, seq: str) -> int:
        ini_file = os.path.join(self.gt_fol, seq, "seqinfo.ini")
        if not os.path.isfile(ini_file):
            raise TrackEvalException(f"ini file does not exist: {seq}/seqinfo.ini")
        ini_data = configparser.ConfigParser()
        ini_data.read(ini_file)
        return int(ini_data["Sequence"]["seqLength"])

    def get_raw_seq_data(self, tracker: str, seq: str) -> dict[str, Any]:
        gt_file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
        pred_file = os.path.join(
            self.tracker_fol, tracker, self.tracker_sub_fol, f"{seq}.txt"
        )
        return {
            "seq": seq,
            "num_timesteps": self.seq_lengths[seq],
            "gt": load_motchallenge_gt(gt_file),
            "pred": load_motchallenge(pred_file),
        }

    def get_preprocessed_seq_data(
        self, raw_data: dict[str, Any], cls: str
    ) -> dict[str, Any]:
        if cls not in self.class_list:
            raise TrackEvalException(f"Class {cls} is not evaluated for this dataset.")
        gt_ids, gt_dets, tracker_ids, tracker_dets = preprocess_motchallenge(
            raw_data["gt"], raw_data["pred"], benchmark=self.benchmark
        )
        return {
            "seq": raw_data["seq"],
            "num_timesteps": raw_data["num_timesteps"],
            "gt_ids": gt_ids,
            "gt_dets": gt_dets,
            "tracker_ids": tracker_ids,
            "tracker_dets": tracker_dets,
            "num_gt_dets": sum(len(f) for f in gt_ids),
            "num_tracker_dets": sum(len(f) for f in tracker_ids),
            "num_gt_ids": len({i for f in gt_ids for i in f}),
            "num_tracker_ids": len({i for f in tracker_ids for i in f}),
        }
