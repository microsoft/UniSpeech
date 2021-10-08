# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, tasks, utils, pdb
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer
from fairseq.models.wav2vec.wav2vec2_s1 import Wav2Vec2Model


@register_model("unispeech")
class Unispeech(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        Wav2Vec2Model.add_args(parser)
        parser.add_argument("--replace-prob", type=float, default=0.5)

    def __init__(self, w2v_encoder, args):
        super().__init__()
        self.w2v_encoder = w2v_encoder
        self.args = args

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        w2v_encoder = Wav2VecEncoder(args, task.target_dictionary, task.source_dictionary)
        return cls(w2v_encoder, args)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x

    # def max_positions(self):
    #     return None


class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, args, tgt_dict=None, src_dict=None):
        self.apply_mask = args.apply_mask

        super().__init__(src_dict)

        self.w2v_model = Wav2Vec2Model.build_model(args) 

        self.final_dropout = nn.Dropout(args.final_dropout)
        self.freeze_finetune_updates = args.freeze_finetune_updates
        self.num_updates = 0
        self.replace_prob = args.replace_prob

        d = args.encoder_embed_dim
        self.encoder_dim = args.encoder_embed_dim

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(args, 'decoder_embed_dim', d) != d:
            self.proj = Linear(d, args.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        contrastive_res = self.w2v_model(**w2v_args)
        x = contrastive_res["features"]
        padding_mask = contrastive_res["padding_mask"]
        q = contrastive_res["q"]

        if tbc:
            x = x.transpose(0, 1)
            q = q.transpose(0, 1)
        replace_mat = torch.empty(x.size(0), x.size(1)).fill_(self.replace_prob)
        replace_mat = torch.bernoulli(replace_mat).bool().to(x.device)
        replace_mat = replace_mat.unsqueeze(-1)
        x = x.masked_fill(replace_mat, 0.0) + q.masked_fill(~replace_mat, 0.0)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "contrastive_res": contrastive_res,
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("unispeech", "unispeech")
def base_architecture(args):
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.final_dropout = getattr(args, "final_dropout", 0)
    args.apply_mask = getattr(args, "apply_mask", True)
    args.dropout = getattr(args, "dropout", 0)
    args.attention_dropout = getattr(args, "attention_dropout", 0)
    args.activation_dropout = getattr(args, "activation_dropout", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.5)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)

    args.freeze_finetune_updates = getattr(args, "freeze_finetune_updates", 0)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0)
    args.layerdrop = getattr(args, "layerdrop", 0.0)
    args.replace_prob = getattr(args, "replace_prob", 0.5)


