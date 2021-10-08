# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import re
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.hubert_criterion import HubertCriterion, HubertCriterionConfig
from fairseq.criterions.ctc import CtcCriterion, CtcCriterionConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round



@dataclass
class UniHubertCriterionConfig(HubertCriterionConfig, CtcCriterionConfig):
    mtlalpha: float = field(
        default=0.5
    )
    
@register_criterion("unihubert", dataclass=UniHubertCriterionConfig)
class UniHubertCriterion(FairseqCriterion):
    def __init__(self, cfg: UniHubertCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.hubert_criterion = HubertCriterion(task, cfg.pred_masked_weight, cfg.pred_nomask_weight, cfg.loss_weights, cfg.log_keys)
        self.ctc_criterion = CtcCriterion(cfg, task)
        self.mtlalpha = cfg.mtlalpha

    def forward(self, model, sample, reduce=True):
        task = sample["task"]
        logging_output = {}
        if task == "multitask":
            hubert_sample = dict()
            hubert_sample['id'] = sample['id']
            hubert_sample['net_input'] = sample['net_input']
            hubert_sample['target_list'] = sample['target_list'][:-1]
        else:
            hubert_sample = sample
        enc_output = self.hubert_criterion.get_net_output(model.w2v_encoder.w2v_model, hubert_sample)
        hubert_loss, hubert_sample_size, hubert_logging_output = self.hubert_criterion.get_loss(
                model.w2v_encoder.w2v_model, sample, enc_output, reduce)

        if task == "multitask":
            ctc_sample = dict()
            ctc_sample['id'] = sample['id']
            ctc_sample['net_input'] = {"source": enc_output['x'], "padding_mask": enc_output["padding_mask"]}
            ctc_sample['target'] = sample['target_list'][-1]
            ctc_loss, ctc_sample_size, ctc_logging_output = self.ctc_criterion(
                    model, ctc_sample, reduce)
        else:
            ctc_loss = 0
            ctc_logging_output = {}
        loss = self.mtlalpha * ctc_loss + (1.0 - self.mtlalpha) * hubert_loss

        logging_output = {"loss": loss, "ntokens": hubert_logging_output["ntokens"],
                "nsentences": hubert_logging_output["nsentences"],
                "hubert": hubert_logging_output, "ctc": ctc_logging_output}

        return loss, hubert_sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        metrics.log_scalar(
            "loss", loss_sum, 1, round=3
        )

        ctc_loss_sum = utils.item(sum(log['ctc'].get('loss', 0) for log in logging_outputs))
        ctc_sample_size = utils.item(sum(log['ctc'].get('sample_size', 0) for log in logging_outputs))
        ctc_ntokens = utils.item(sum(log['ctc'].get('ntokens', 0) for log in logging_outputs))
        ctc_nsentences = utils.item(sum(log['ctc'].get('nsentences', 0) for log in logging_outputs))

        hubert_loss_sum = utils.item(sum(log['hubert'].get('loss', 0) for log in logging_outputs))
        hubert_sample_size = utils.item(sum(log['hubert'].get('sample_size', 0) for log in logging_outputs))
        hubert_ntokens = utils.item(sum(log['hubert'].get('ntokens', 0) for log in logging_outputs))
        hubert_nsentences = utils.item(sum(log['hubert'].get('nsentences', 0) for log in logging_outputs))


        if ctc_sample_size != 0:
            metrics.log_scalar(
                "ctc_loss", ctc_loss_sum / ctc_sample_size / math.log(2), ctc_sample_size, round=3
            )

        if hubert_sample_size != 0:
            metrics.log_scalar(
                "hubert_loss", hubert_loss_sum/ hubert_sample_size / math.log(2), hubert_sample_size, round=3
            )
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))
            counts = {}
            for lk in logging_outputs[0]["hubert"].keys():
                if lk.startswith("count_"):
                    val = sum(log["hubert"][lk] for log in logging_outputs)
                    metrics.log_scalar(lk, val)
                    counts[lk] = val

            for lk in logging_outputs[0]['hubert'].keys():
                if lk.startswith("loss_"):
                    val = sum(log["hubert"][lk] for log in logging_outputs)
                    metrics.log_scalar(lk, val / hubert_sample_size / math.log(2), round=3)
                elif lk.startswith("correct_"):
                    val = sum(log["hubert"][lk] for log in logging_outputs)
                    metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])


        c_errors = sum(log['ctc'].get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log['ctc'].get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log['ctc'].get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log['ctc'].get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log['ctc'].get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)


        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

        metrics.log_scalar("hubert_nsentences", hubert_nsentences)
        metrics.log_scalar("hubert_sample_size", hubert_sample_size)
        metrics.log_scalar("ctc_nsentences", ctc_nsentences)
        metrics.log_scalar("ctc_sample_size", ctc_sample_size)

        correct = sum(log["hubert"].get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)
        total = sum(log['hubert'].get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
