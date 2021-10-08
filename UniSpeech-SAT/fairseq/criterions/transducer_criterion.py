# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import LegacyFairseqCriterion, register_criterion
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import safe_round
from fairseq.criterions.rnnt import RNNTLoss


@register_criterion("transducer")
class TransducerCriterion(LegacyFairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.blank_idx = task.target_dictionary.bos()
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.sentence_avg = args.sentence_avg
        self.criterion = RNNTLoss()


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        return

    def get_net_output(self, model, sample):
        net_output = model(**sample)
        return net_output

    def get_loss(self, model, sample, net_output, reduce=True):
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"].int()
        else:
            non_padding_mask = ~net_output["padding_mask"]
            input_lengths = non_padding_mask.int().sum(-1)

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        target_lengths = sample["target_lengths"]

        loss_scale = sample['loss_scale'] if 'loss_scale' in sample else 1.0


        with torch.backends.cudnn.flags(enabled=False):
            loss = self.criterion(
                lprobs,
                targets_flat,
                input_lengths.int(),
                target_lengths.int(),
                blank=self.blank_idx,
                loss_scale=loss_scale)

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        net_output = self.get_net_output(model, sample)
        loss, sample_size, logging_output = self.get_loss(model, sample, net_output, reduce)

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens, ntokens, round=3
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
