# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import pdb
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from fairseq.data import Dictionary, HubertDataset, MultitaskDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False,
        )


@dataclass
class UniHubertPretrainingConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help": "path to data directory"}
    )
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    sample_ratios: List[float] = field(
            default_factory=lambda: [1.0, 1.0]
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: int = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
    )
    sample_rate: int = field(
        default=16000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={
            "help": "if set, normalizes input to have 0 mean and unit variance"
        },
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys "
            "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )


@register_task("unihubert_pretraining", dataclass=UniHubertPretrainingConfig)
class UniHubertPretrainingTask(FairseqTask):

    cfg: UniHubertPretrainingConfig

    def __init__(
        self,
        cfg: UniHubertPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"HubertPretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning
        self.inner_update = 0

        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)

        self._source_dictionary = None

        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return self._source_dictionary

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        if hasattr(self.state, "target_dictionary"):
            return self.state.target_dictionary
        else:
            return self.state.dictionaries[-1]

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries

    @classmethod
    def setup_task(
        cls, cfg: UniHubertPretrainingConfig, **kwargs
    ) -> "HubertPretrainingTask":
        return cls(cfg)

    def sampling_func(self, x):
        if self.inner_update % 2 == 0:
            return x[0]
        return x[1]


    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [Dictionary.load(f"{label_dir}/dict.{label}.txt") for label in self.cfg.labels]
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split: str, **kwargs) -> None:
        splits = split.split(",")
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        if len(splits) == 2:
            manifest_list = [f"{self.cfg.data}/{subset}.tsv" for subset in splits]
            unsup_paths = [f"{self.get_label_dir()}/{splits[0]}.{l}" for l in self.cfg.labels[:-1]]

            sup_paths = [
                f"{self.get_label_dir()}/{splits[1]}.{l}" for l in self.cfg.labels
            ]
            unsup_dataset = HubertDataset(
                manifest_list[0],
                sample_rate=self.cfg.sample_rate,
                label_paths=unsup_paths,
                label_rates=self.cfg.label_rate,
                pad_list=pad_list[:-1],
                eos_list=eos_list[:-1],
                label_processors=procs[:-1],
                max_keep_sample_size=None,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                single_target=False,
            )

            sup_dataset = HubertDataset(
                manifest_list[1],
                sample_rate=self.cfg.sample_rate,
                label_paths=sup_paths,
                label_rates=[self.cfg.label_rate,-1],
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                max_keep_sample_size=None,
                min_keep_sample_size=None,
                max_sample_size=None,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                single_target=False,
                multitask=True
            )
            datasets = OrderedDict()
            datasets["unsup"] = unsup_dataset
            datasets["sup"] = sup_dataset


            self.datasets[split] = MultitaskDataset(
                    [unsup_dataset, sup_dataset], self.cfg.sample_ratios
            )
        else:
            manifest = f"{self.cfg.data}/{split}.tsv" 
            paths = [
                    f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels
            ]
            self.datasets[split] = HubertDataset(
                manifest,
                sample_rate=self.cfg.sample_rate,
                label_paths=paths,
                label_rates=[self.cfg.label_rate, -1],
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                max_keep_sample_size=None,
                min_keep_sample_size=None,
                max_sample_size=None,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                single_target=False,
                multitask=True
            )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self, indices: np.array, *args, **kwargs
    ) -> np.array:
        return indices
