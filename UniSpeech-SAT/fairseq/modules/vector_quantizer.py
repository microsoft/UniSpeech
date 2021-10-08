#####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 # Copyright (C) 2018 Zalando Research                                               #
 #                                                                                   #
 # This file is part of VQ-VAE-Speech.                                               #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations, product


class VectorQuantizer(nn.Module):
    """
    Inspired from Sonnet implementation of VQ-VAE https://arxiv.org/abs/1711.00937,
    in https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py and
    pytorch implementation of it from zalandoresearch in https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb.
    Implements the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937
    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.
    The output tensor will have the same shape as the input.
    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.
    Args:
        embedding_dim: integer representing the dimensionality of the tensors in the
            quantized space. Inputs to the modules must be in this format as well.
        num_embeddings: integer, the number of vectors in the quantized space.
            commitment_cost: scalar which controls the weighting of the loss terms
            (see equation 4 in the paper - this variable is Beta).
    """
    
    def __init__(
        self,
        dim,
        num_vars,
        vq_dim,
        time_first,
        gamma
        ):
        super(VectorQuantizer, self).__init__()

        self.input_dim = dim
        self.num_vars = num_vars
        self.vq_dim = vq_dim
        self.time_first = time_first
        self.gamma = gamma

        self.weight_proj = nn.Linear(self.input_dim, vq_dim)

        self.vars = nn.Parameter(torch.FloatTensor(num_vars, vq_dim))
        self.vars.data.uniform_(-1/num_vars, 1/num_vars)

    def forward(self, x, produce_targets=False):
        """
        Connects the module to some inputs.
        Args:
            inputs: Tensor, final dimension must be equal to embedding_dim. All other
                leading dimensions will be flattened and treated as a large batch.
        Returns:
            loss: Tensor containing the loss to optimize.
            quantize: Tensor containing the quantized version of the input.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
                of the quantized space each input element was mapped to.
            distances
        """

        result = {"num_vars": self.num_vars}

        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz, -1)

        x_expand = x.unsqueeze(-1)
        distance = torch.norm(x_expand - self.vars.t(), 2, 1)
        distance = distance.view(bsz * tsz, self.num_vars)   # (bsz * tsz,  n_var)
            
        k = torch.argmin(distance, dim=1)    # (bsz * tsz)

        hard_x = (
            distance.new_zeros(*distance.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, -1)           # (bsz * tsz, n_var)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()
        import pdb
        pdb.set_trace()

        quantized = torch.matmul(hard_x.view(bsz * tsz, -1), self.vars)  # (bsz * tsz, group * n_vars) (group * n_var, var_dim)

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(x.detach(), quantized)
        quantized = x + (quantized - x).detach()
        result["latent_loss"] = self.gamma * e_latent_loss + q_latent_loss
        result["encoding_indices"] = k
        result["quantized"] = quantized.view(bsz, tsz, -1).permute(0, 2, 1).contiguous()
      
        return result

        
