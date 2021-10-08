# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import pdb
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class VqvaeCriterionConfig(FairseqDataclass):
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )

@register_criterion("vqvae", dataclass=VqvaeCriterionConfig)
class VqvaeCriterion(FairseqCriterion):
    def __init__(self, task, infonce=False, loss_weights=None, log_keys=None):
        super().__init__(task)
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, sample, reduce=True):
        results = model(sample["input"], sample["target"], reduce=reduce)
        y = results["y"]
        target = sample["target"]
        length = y.shape[1]
        if target.shape[1] > length:
            target = target[:, :length, :]
        recon_loss = F.mse_loss(
            y,
            target
        )
        losses = []

        sample_size = y.shape[0] * y.shape[1]
        loss = recon_loss * sample_size
        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(results) 
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)
        
        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        for lk in self.log_keys:
            if lk in results:
                logging_output[lk] = float((results[lk]))

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item()

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
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)


        builtin_keys = {
            "loss",
            "ntokens",
            "nsentences",
            "sample_size"
        }

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs) / len(
                    logging_outputs
                )
                if k.startswith("loss"):
                    metrics.log_scalar(k, val / sample_size / math.log(2), sample_size)
                else:
                    metrics.log_scalar(k, val, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
