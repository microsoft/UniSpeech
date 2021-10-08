"""Transducer for training and decoding."""

import six
import pdb

import torch
import torch.nn.functional as F
import numpy as np
import copy, math
from fairseq.models import FairseqDecoder

def log_aplusb(a, b):
    """
    implement log(a + b)
    """
    return max(a, b) + math.log1p(math.exp(-math.fabs(a - b)))

class Sequence:
    def __init__(self, seq=None, hidden=None, blank=0):
        if seq is None:
            self.g = []  # predictions of phoneme language model
            self.k = [blank]  # prediction phoneme label
            self.h = hidden
            self.logp = 0  # probability of this sequence, in log scale
        else:
            self.g = seq.g[:]  # save for prefixsum
            self.k = seq.k[:]
            self.h = seq.h
            self.logp = seq.logp


class LinearND(torch.nn.Module):

    def __init__(self, *args, use_weight_norm=True):
        """
        A torch.nn.Linear layer modified to accept ND arrays.
        The function treats the last dimension of the input
        as the hidden dimension.
        """
        super(LinearND, self).__init__()
        self.fc = torch.nn.Linear(*args)
        if use_weight_norm:
            self.fc = torch.nn.utils.weight_norm(self.fc, name='weight')

    def forward(self, x):
        #print(x.shape)
        size = x.size()

        n = np.prod(size[:-1])
        out = x.contiguous().view(int(n), size[-1])
        #print(out.shape)
        out = self.fc(out)
        size = list(size)
        size[-1] = out.size()[-1]
        #print(size)
        return out.view(size)


class LinearND_Lnorm(torch.nn.Module):

    def __init__(self, *args):
        """
        A torch.nn.Linear layer modified to accept ND arrays.
        The function treats the last dimension of the input
        as the hidden dimension.
        """
        super(LinearND_Lnorm, self).__init__()
        self.fc = torch.nn.Linear(*args)
        self.Lnorm = torch.nn.LayerNorm(args[1])

    def forward(self, x):
        out = self.fc(x)
        return self.Lnorm(out)

class LSTMNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, layer_norm=True):
        super().__init__()
        self.lnorm_layer = None
        self.dropout_layer = None
        self.rnn = torch.nn.LSTM(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=False)
        if layer_norm:
            self.lnorm_layer = torch.nn.LayerNorm(hidden_size)

        if dropout > 0:
            self.dropout_layer = torch.nn.Dropout(p=dropout)

    def forward(self, x, h0=None):
        if h0 is None:
            #print(x.shape)
            out, hidden = self.rnn(x)
            #print(hidden[0].shape,hidden[1].shape)
        else:
            #print(x.shape, h0[0].shape,h0[1].shape)
            out, hidden = self.rnn(x, h0)

        if self.dropout_layer is not None:
            out = self.dropout_layer(out)
        if self.lnorm_layer is not None:
            out = self.lnorm_layer(out)

        return out, hidden

class RNNTDecoder(FairseqDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(tgt_dict)

        odim = len(tgt_dict)

        self.embed = torch.nn.Embedding(odim, args.decoder_embed_dim, padding_idx=tgt_dict.pad())
        if args.embed_layer_norm:
            self.embed_lnorm = torch.nn.LayerNorm(args.decoder_embed_dim)
        else:
            self.embed_lnorm = None

        if args.embed_sigmoid:
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self.sigmoid = None

        self.decoder = LSTMNetwork(args.decoder_embed_dim, args.decoder_dim, args.decoder_layers, args.dropout)

        if args.linear_norm:
            self.lin_enc = LinearND_Lnorm(args.encoder_proj_dim, args.joint_dim)
            self.lin_dec = LinearND_Lnorm(args.decoder_dim, args.joint_dim)
        else:
            self.lin_enc = LinearND(args.encoder_proj_dim, args.joint_dim, args.weight_norm)
            self.lin_dec = LinearND(args.decoder_dim, args.joint_dim, args.weight_norm)

        self.lin_out = LinearND(args.joint_dim, odim, args.weight_norm)

        self.dlayers = args.decoder_layers
        self.dunits = args.decoder_dim
        self.embed_dim = args.decoder_embed_dim
        self.joint_dim = args.joint_dim
        self.odim = odim
        self.blank = tgt_dict.bos()
        self.pad = tgt_dict.pad()

    def joint(self, h_enc, h_dec):
        z = torch.relu(self.lin_enc(h_enc) + self.lin_dec(h_dec))
        z = self.lin_out(z)

        return z

    def forward(self, hs_pad, ys_in_pad):
        ys_in_pad = torch.nn.functional.pad(ys_in_pad, [1,0], value=self.pad).long()
        eys = self.embed(ys_in_pad)
        if self.embed_lnorm is not None:
            eys = self.embed_lnorm(eys)
        if self.sigmoid is not None:
            eys = self.sigmoid(eys)

        h_dec, _ = self.decoder(eys, None)

        h_enc = hs_pad.unsqueeze(2)
        h_dec = h_dec.unsqueeze(1)

        z = self.joint(h_enc, h_dec)
        return z

    def beam_search(self, h, beam_size, prefix=True):
            '''
            `xs`: acoustic model outputs
            NOTE only support one sequence (batch size = 1)
            including dummy parameters sos, eos for Seq2Seq compatibility
            '''

            def forward_step(label, hidden):
                y = torch.tensor([label], dtype=torch.int64, device='cuda')
                y = self.embed(y)
                if self.embed_lnorm is not None:
                    y = self.embed_lnorm(y)
                if self.sigmoid is not None:
                    y = self.sigmoid(y)
                y = y.unsqueeze(0)
                y_out, hidden = self.decoder(y, hidden)
                return y_out, hidden

            def isprefix(a, b):
                # a is the prefix of b
                if a == b or len(a) >= len(b):
                    return False
                for i in range(len(a)):
                    if a[i] != b[i]:
                        return False
                return True

            def merge_prefix(A, f1):
                for j in range(len(A) - 1):
                    for i in range(j + 1, len(A)):
                        if not isprefix(A[i].k, A[j].k):
                            continue
                        # A[i] -> A[j]
                        # TODO: group these calls to batch?
                        y_out, _ = forward_step(A[i].k[-1], A[i].h)
                        y_out = self.lin_dec(y_out[0, 0, :])
                        ytu = torch.log_softmax(self.lin_out(torch.relu(f1+y_out)), dim=0)
                        ytu = ytu.cpu().numpy()

                        idx = len(A[i].k)
                        curlogp = A[i].logp + float(ytu[A[j].k[idx]])

                        # TODO: no need for loop here, can run in parallel
                        for k in range(idx, len(A[j].k) - 1):
                            f2 = A[j].g[k]
                            out = f1 + f2
                            out = torch.log_softmax(self.lin_out(torch.relu(out)), dim=0)
                            logp = out.data.cpu().numpy()
                            curlogp += float(logp[A[j].k[k + 1]])
                        A[j].logp = log_aplusb(A[j].logp, curlogp)

            def recalculate_beam(A, B, f1, beam):
                # A is no longer used after this function, so we will
                # restructure everything for faster data access

                # to optimize beam recalculation we will never construct
                # a candidate (i.e. Sequence class), unless it is actually
                # added to next beam (i.e. B list)
                decoder_time = 0.0
                logprobs = [np.array([a.logp for a in A])]
                # TODO: think of removing prev_probs as this is redundant list
                prev_probs = [0]
                hidden_states = []
                if prefix:
                    output_states = []
                best_itr = 0
                best_idx = logprobs[0].argmax()
                y_hat = A[best_idx]
                prev_seqs = []
                while True:
                    # instead of removing element, let's make it
                    # highly unlikely to avoid costly data copy
                    logprobs[best_itr][best_idx] = -1e9

                    y_out, hidden = forward_step(y_hat.k[-1], y_hat.h)
                    y_out = self.lin_dec(y_out[0][0])
                    ytu = torch.log_softmax(self.lin_out(torch.relu(f1+y_out)), dim=0)
                    ytu = ytu.cpu().numpy()

                    # saving data to use in delayed computation
                    prev_seqs.append(y_hat)
                    prev_probs.append(y_hat.logp)
                    # TODO: exclude blank properly
                    if prefix:
                        output_states.append(y_out)

                    # adding current best candidate to B
                    yk = Sequence(y_hat)
                    yk.logp += ytu[self.blank]

                    flag_merge = True

                    for y_b in B:
                        if yk.k == y_b.k:
                            if prefix:
                                y_b.logp = max(y_b.logp, yk.logp)
                            else:
                                y_b.logp = log_aplusb(y_b.logp, yk.logp)
                            flag_merge = False
                            break
                    if flag_merge:
                        B.append(yk)

                    ytu[self.blank] = -1e9
                    logprobs.append(ytu)
                    hidden_states.append(hidden)

                    # finding best probability by recalculating
                    # from saved results
                    best_itr = 0
                    best_idx = 0
                    best_logp = -1e9
                    for itr, itr_log_probs in enumerate(logprobs):
                        itr_best_idx = itr_log_probs.argmax()
                        itr_best_logp = itr_log_probs[itr_best_idx] + prev_probs[itr]
                        if itr_best_logp > best_logp:
                            best_itr = itr
                            best_idx = itr_best_idx
                            best_logp = itr_best_logp

                    if best_itr == 0:
                        y_hat = A[best_idx]
                    else:
                        # constructing new y_hat by recalculating
                        # from saved results
                        y_hat = Sequence(prev_seqs[best_itr - 1])
                        y_hat.logp = best_logp
                        y_hat.h = hidden_states[best_itr - 1]
                        if prefix:
                            y_hat.g.append(output_states[best_itr - 1])
                        y_hat.k.append(best_idx.item())

                    B = sorted(B, key=lambda a: a.logp, reverse=True)
                    if len(B) >= beam and B[beam - 1].logp >= y_hat.logp:
                        return B

            decoder_time = 0.0
            beam = beam_size
            preds = h
            T, V = preds.shape

            h0 = torch.zeros([self.dlayers, 1, self.dunits], device="cuda")
            h0 = (h0, h0)
            xs = self.lin_enc(preds)

            n_best = []
            n_best_model_scores = []
            n_best_final_scores = []
            results = []
            # TODO: use batch size > 1, to parallelize encoder computation
            B = [Sequence(blank=self.blank, hidden=h0)]
            for t in range(len(xs)):
                f1 = xs[t]
                A = B
                B = []
                if prefix:
                    # larger sequence first add
                    A = sorted(A, key=lambda a: len(a.k), reverse=True)
                    merge_prefix(A, f1)
                B = recalculate_beam(A, B, f1, beam)
                Bsort = sorted(B, key=lambda a: a.logp, reverse=True)
                B = Bsort[:beam]

            for ci in B:
                if len(ci.k) - 1 > 0:
                    ci.logp_lengthnorm = ci.logp / (len(ci.k)-1)
                else:
                    ci.logp_lengthnorm = ci.logp

            Bsort = sorted(B, key=lambda a: a.logp_lengthnorm, reverse=True)

            for i in range(len(Bsort)):
                n_best.append([])
                n_best_model_scores.append(Bsort[i].logp)
                n_best_final_scores.append(Bsort[i].logp_lengthnorm)
                for c in Bsort[i].k:
                    n_best[-1].append(c)

            hyp = {'tokens': n_best[0], 'score': n_best_final_scores[0]}

            return [hyp]
