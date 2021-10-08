# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import editdistance
import os
import sys
import numpy as np
import logging
import pdb
import torch

from dataclasses import dataclass, field
from typing import Optional, Any
from fairseq.data import AddTargetDataset, Dictionary, ChunkAudioDataset, ResamplingDataset, ConcatDataset, encoders, FairseqDataset, iterators
from fairseq.data.data_utils import post_process
from fairseq.tasks.audio_pretraining import LabelEncoder, IDEncoder, AudioPretrainingTask, AudioPretrainingConfig

from . import FairseqTask, register_task
from .. import utils
from ..logging import metrics


logger = logging.getLogger(__name__)


@dataclass
class ChunkAudioPretrainingConfig(AudioPretrainingConfig):
    train_chunk_files: str = field(
        default="file_set.json",
        metadata={"help": "comma separated list of data subsets to use for validation"},
    )
    train_chunk_paths: Optional[str] = field(
        default=None, metadata={"help": "chunk paths"},
    )
    train_trans_paths: Optional[str] = field(
        default=None
    )
    train_subcorpus: Optional[str] = field(
        default=None)
    valid_chunk_files: str = field(
        default="file_set.json",
        metadata={"help": "comma separated list of data subsets to use for validation"},
    )
    valid_chunk_paths: Optional[str] = field(
        default=None, metadata={"help": "chunk paths"},
    )
    valid_trans_paths: Optional[str] = field(
        default=None
    )
    valid_subcorpus: Optional[str] = field(
        default=None
    )

    test_chunk_files: str = field(
        default="file_set.json"
    )
    test_chunk_paths: Optional[str] = field(
        default=None,
    )
    test_trans_paths: Optional[str] = field(
        default=None
    )
    test_subcorpus: Optional[str] = field(
        default=None
    )
    feature: Optional[str] = field(
        default="audio"
    )
    mean_file: Optional[str] = field(
        default=None
    )
    invstd_file: Optional[str] = field(
        default=None
    )
    batch_criterion: Optional[str] = field(
        default="frame"
    )


@register_task("chunk_audio_pretraining", dataclass=ChunkAudioPretrainingConfig)
class ChunkAudioPretrainingTask(AudioPretrainingTask):
    """"""
    def __init__(self, cfg: ChunkAudioPretrainingConfig):
        super().__init__(cfg)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (omegaconf.DictConfig): parsed command-line arguments
        """

        return cls(cfg)


    def load_dataset(self, split, task_cfg=None, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        task_cfg = task_cfg or self.cfg
        if "train" in split:
            manifest_list = self.cfg.train_chunk_files.split(",")
            chunk_paths = self.cfg.train_chunk_paths.split(",") if self.cfg.train_chunk_paths else [None] * len(manifest_list)
            trans_paths = self.cfg.train_trans_paths.split(",") if self.cfg.train_trans_paths else chunk_paths
            subset = self.cfg.train_subcorpus.split(",") if self.cfg.train_subcorpus else [None] * len(manifest_list)
        elif "valid" in split:
            manifest_list = self.cfg.valid_chunk_files.split(",")
            chunk_paths = self.cfg.valid_chunk_paths.split(",") if self.cfg.valid_chunk_paths else [None] * len(manifest_list)
            trans_paths = self.cfg.valid_trans_paths.split(",") if self.cfg.valid_trans_paths else chunk_paths
            subset = self.cfg.valid_subcorpus.split(",") if self.cfg.valid_subcorpus else [None] * len(manifest_list)
        elif "test" in split:
            manifest_list = self.cfg.test_chunk_files.split(",")
            chunk_paths = self.cfg.test_chunk_paths.split(",") if self.cfg.test_chunk_paths else [None] * len(manifest_list)
            trans_paths = self.cfg.test_trans_paths.split(",") if self.cfg.test_trans_paths else chunk_paths
            subset = self.cfg.test_subcorpus.split(",") if self.cfg.test_subcorpus else [None] * len(manifest_list)


        datasets = []
        datasets_lengths = []
        for i in range(len(manifest_list)):
            dataset = ChunkAudioDataset(
                manifest_list[i],
                chunk_paths[i],
                trans_paths[i],
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                max_tokens=task_cfg.max_tokens,
                pad=self.cfg.labels is not None or self.cfg.enable_padding,
                normalize=self.cfg.normalize,
                subset=subset[i],
                shuffle='train' in split,
                shard='train' in split,
                label=self.cfg.labels,
                dictionary=self.target_dictionary,
                feature=self.cfg.feature,
                mean_file=self.cfg.mean_file,
                invstd_file=self.cfg.invstd_file,
                batch_criterion=task_cfg.batch_criterion
            )

            datasets.append(dataset)
            datasets_lengths.append(len(dataset))

        if len(manifest_list) == 1:
            self.datasets[split] = dataset
        else:
            languages = [manifest.split('/')[0] for manifest in manifest_list]
            datasets_lengths = np.array(datasets_lengths)
            sample_probs = self._get_sample_prob(datasets_lengths)
            for id, lang in enumerate(languages):
                logger.info(
                    "Sample probability by language: {} : {:.4f}".format(lang, sample_probs[id])
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
                    seed=self.cfg.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(datasets)
            ]
            self.datasets[split] = ConcatDataset(resampled_lang_datasets)


    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochPipeIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            seed=seed,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
        )

        return epoch_iter


    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.)
        num_char_errors = sum(log.get("_num_char_errors", zero) for log in logging_outputs)
        num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
        num_word_errors = sum(log.get("_num_word_errors", zero) for log in logging_outputs)
        num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
        metrics.log_scalar("_num_char_errors", num_char_errors)
        metrics.log_scalar("_num_chars", num_chars)
        metrics.log_scalar("_num_word_errors", num_word_errors)
        metrics.log_scalar("_num_words", num_words)
        if num_words > 0:
            metrics.log_derived(
                "uer",
                lambda meters: meters["_num_char_errors"].sum * 100.0 / meters["_num_chars"].sum
                if meters["_num_chars"].sum > 0 else float("nan")
            )
            metrics.log_derived(
                "wer",
                lambda meters: meters["_num_word_errors"].sum * 100.0 / meters["_num_words"].sum
                if meters["_num_words"].sum > 0 else float("nan")
            )
