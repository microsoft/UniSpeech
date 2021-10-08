# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING, II, OmegaConf, dictconfig

from fairseq.data import (
    ConcatDataset,
    FeatsAudioDataset,
    ResamplingDataset,
    encoders,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from . import FairseqTask, register_task
from .. import utils
from ..logging import metrics


logger = logging.getLogger(__name__)

@dataclass
class VqvaeTaskConfig(FairseqDataclass):
    data: Optional[str] = field(default=MISSING, metadata={"help": "path to data directory"})
    train_path: Optional[str] = field(default=MISSING)

    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    input_feature: str = field(
        default="mfcc",
        metadata={"help": "input feature for vqvae model"}
    )
    output_feature: str = field(
        default="mfcc",
        metadata={"help": "target feature for vqvae model"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )


@register_task("vqvae", dataclass=VqvaeTaskConfig)
class VqvaeTask(FairseqTask):
    """ """

    cfg: VqvaeTaskConfig

    def __init__(
        self,
        cfg: VqvaeTaskConfig,
    ):
        super().__init__(cfg)

    @classmethod
    def setup_task(cls, cfg: VqvaeTaskConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (VqvaeTaskConfig): configuration of this task
        """

        return cls(cfg)

    @property
    def target_dictionary(cls):
        return None

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        manifest_list = split.split(",")
        datasets = []
        datasets_lengths = []

        for f in manifest_list:
            manifest = os.path.join(self.cfg.data, "{}.tsv".format(f))
            dataset = FeatsAudioDataset(
                manifest,
                sample_rate=task_cfg.sample_rate,
                input_feature=task_cfg.input_feature,
                output_feature=task_cfg.output_feature,
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                normalize=task_cfg.normalize,
            )


        if len(manifest_list) == 1:
            self.datasets[split] = dataset
        else:
            languages = [manifest.split('/')[0] for manifest in manifest_list]
            datasets_lengths = np.array(datasets_lengths)
            sample_probs = self._get_sample_prob(datasets_lengths)
            for id, lang in enumerate(languages):
                logger.info(
                    "Sample probability by language: {} : {:.5f}".format(lang, sample_probs[id])
                 )
            size_ratio = (sample_probs * datasets_lengths.sum()) / datasets_lengths
            for id, lang in enumerate(languages):
                logger.info(
                    "Up/Down Sampling ratio by language: {} : {:.2f}".format(lang, size_ratio[id])
                )
            resampled_lang_datasets = [
                ResamplingDataset(
                    datasets[i],
                    size_ratio=size_ratio[i],
                    seed=task_cfg.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(datasets)
            ]
            self.datasets[split] = ConcatDataset(resampled_lang_datasets)


    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)

        actualized_cfg = getattr(model, "cfg", None)

        return model
    
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

