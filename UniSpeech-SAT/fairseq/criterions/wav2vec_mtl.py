import torch
import math

from fairseq import pdb
from fairseq import utils, metrics
from fairseq.criterions import LegacyFairseqCriterion, register_criterion
from fairseq.criterions.wav2vec_criterion import Wav2vecCriterion
from fairseq.criterions.ctc import CtcCriterion
from fairseq.logging.meters import safe_round

@register_criterion('wav2vec_mtl')
class Wav2vecMTLCriterion(LegacyFairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.mtlalpha = args.mtlalpha
        self.w2v_criterion = Wav2vecCriterion(task, args.infonce, args.loss_weights, args.log_keys)
        if self.mtlalpha > 0:
            self.ctc_criterion = CtcCriterion(args, task)

    @staticmethod
    def add_args(parser):
        Wav2vecCriterion.add_args(parser)
        CtcCriterion.add_args(parser)
        parser.add_argument('--mtlalpha', type=float, default=0.5)


    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
     
        if self.mtlalpha > 0.0:
            ctc_loss, ctc_sample_size, ctc_logging_output = self.ctc_criterion.get_loss(model, sample, net_output, reduce)
        else:
            ctc_loss = 0
            ctc_sample_size = 0
            ctc_logging_output = {}

        infonce_loss, infonce_sample_size, infonce_logging_output = self.w2v_criterion.get_loss(model.w2v_encoder.w2v_model, sample, net_output['contrastive_res'], reduce)
        loss = self.mtlalpha * ctc_loss + (1.0 - self.mtlalpha) * infonce_loss
        sample_size = infonce_sample_size
        logging_output = {'loss': loss, 'ntokens': ctc_logging_output['ntokens'], 'nsentences': ctc_logging_output['nsentences'],
                           'ctc': ctc_logging_output, 'infonce': infonce_logging_output}

        return loss, sample_size, logging_output

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        ctc_loss_sum = utils.item(sum(log['ctc'].get('loss', 0) for log in logging_outputs))
        ctc_sample_size = utils.item(sum(log['ctc'].get('sample_size', 0) for log in logging_outputs))
        ctc_ntokens = utils.item(sum(log['ctc'].get('ntokens', 0) for log in logging_outputs))
        ctc_nsentences = utils.item(sum(log['ctc'].get('nsentences', 0) for log in logging_outputs))

        ctras_loss_sum = utils.item(sum(log['infonce'].get('loss', 0) for log in logging_outputs)) 
        ctras_sample_size = utils.item(sum(log['infonce'].get('sample_size', 0) for log in logging_outputs))
        ctras_ntokens = utils.item(sum(log['infonce'].get('ntokens', 0) for log in logging_outputs))
        ctras_nsentences = utils.item(sum(log['infonce'].get('nsentences', 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss_sum, 1, round=3)
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / ctc_sample_size / math.log(2), ctc_sample_size, round=3
        )
        metrics.log_scalar(
            "contrastive_loss", ctras_loss_sum / ctras_sample_size / math.log(2), ctras_sample_size, round=3
        )
        if ctc_sample_size != ctc_ntokens:
            metrics.log_scalar(
                "nll_loss", ctc_loss_sum / ctc_ntokens / math.log(2), ntokens, round=3
            )
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
                lambda meters: safe_round(meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3)
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3)
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3)
                if meters["_w_total"].sum > 0
                else float("nan"),
            )


        metrics.log_scalar("nsentences", ctras_nsentences)
        metrics.log_scalar("ctc_sample_size", ctc_sample_size)
        metrics.log_scalar("contrastive_sample_size", ctras_sample_size)

                
        correct = sum(log['infonce'].get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)
  
        total = sum(log['infonce'].get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)


        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(meters["_correct"].sum / meters["_total"].sum, 5)
                if meters["_total"].sum > 0
               else float("nan"),
            )

        builtin_keys = {'loss', 'ntokens', 'nsentences', 'sample_size', 'correct', 'count'}
        for k in logging_outputs[0]['infonce']:
            if k not in builtin_keys:
                val = sum(log['infonce'].get(k, 0) for log in logging_outputs) / len(logging_outputs)
                if k.startswith('loss'):
                    metrics.log_scalar(k, val / ctras_sample_size / math.log(2), ctras_sample_size)
                else:
                    metrics.log_scalar(k, val, round=3)    

