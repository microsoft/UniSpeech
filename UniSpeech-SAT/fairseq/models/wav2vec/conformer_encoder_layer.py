import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
)


class Swish(nn.Module):
    """Swish function
    """

    def __init__(self):
        """Construct an MultiHeadedAttention object."""
        super(Swish, self).__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.act(x)
              
class GLU_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, glu_type = "sigmoid", bias_in_glu=True):
        super(GLU_Linear, self).__init__()        

        self.glu_type = glu_type
        self.output_dim = output_dim
        

        if glu_type == "sigmoid":
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = Swish()
        elif glu_type == "relu":
            self.glu_act = torch.nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = torch.nn.GELU()
            

        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim *2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim *2, False)
          
    def forward(self, x):
       # to be consistent with GLU_Linear, we assume the input always has the #channel (#dim) in the last dimension of the tensor, so need to switch the dimension first for 1D-Conv case
        x = self.linear(x)
        
        if self.glu_type == "bilinear":
            x = (x[:,:,0:self.output_dim] * x[:,:,self.output_dim:self.output_dim*2])
        else:
            x = (x[:, :, 0:self.output_dim] * self.glu_act(x[:, :, self.output_dim:self.output_dim*2]))
            
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout_rate):
        super(FeedForward, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        self.layer_norm = LayerNorm(d_model)
        self.net = nn.Sequential(
            GLU_Linear(d_model, d_inner, "swish", True),
            nn.Dropout(dropout_rate),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout_rate)
        )


    def forward(self, x):
        x = self.layer_norm(x)
        out = self.net(x)

        return out


class ConvModule(nn.Module):
    def __init__(self, input_dim, kernel_size, dropout_rate, causal=False, bn=False):
        super(ConvModule, self).__init__()
        self.layer_norm = LayerNorm(input_dim)

        #self.pw_conv_1 = nn.Conv2d(1, 2, 1, 1, 0)
        #self.pw_conv_2 = nn.Conv2d(1, 1, 1, 1, 0)
        self.glu_act = torch.nn.Sigmoid()
        self.causal = causal
        self.bn = bn
        self.kernel_size = kernel_size

        self.pw_conv_simplify_w = torch.nn.Parameter(torch.ones(3))
        self.pw_conv_simplify_b = torch.nn.Parameter(torch.zeros(3))

        if causal:
            self.dw_conv_1d = nn.Conv1d(input_dim, input_dim, kernel_size, 1, padding=(kernel_size-1), groups=input_dim)
        else:
            self.dw_conv_1d = nn.Conv1d(input_dim, input_dim, kernel_size, 1, padding=(kernel_size-1)//2, groups=input_dim)
        if bn:
            self.BN = nn.BatchNorm1d(input_dim)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.layer_norm(x)
        
        x_0 = x * self.pw_conv_simplify_w[0] + self.pw_conv_simplify_b[0]
        x_1 = x * self.pw_conv_simplify_w[1] + self.pw_conv_simplify_b[1]
        x = x_0 + x_1
        
        #x = self.pw_conv_1(x)
        #x = x[:, 0] * self.glu_act(x[:, 1])
        x = x.permute([0, 2, 1])

        x = self.dw_conv_1d(x)
        if self.causal:
            x = x[:, :, :-(self.kernel_size-1)]
        if self.bn:
            x = self.BN(x)
        x = self.act(x)
        x = x.unsqueeze(1).permute([0, 1, 3, 2])
        #x = self.pw_conv_2(x)
        
        x = x * self.pw_conv_simplify_w[2] + self.pw_conv_simplify_b[2]
        x = self.dropout(x).squeeze(1)

        return x



class ConformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 0,
        max_distance: int = 0,
        rescale_init: bool=False,
        gru_rel_pos: bool=False,
        expand_attention_head_size: int=-1,
        bn: bool=False

    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.feed_forward_in = FeedForward(self.embedding_dim, ffn_embedding_dim, activation_dropout)
        self.activation_name = activation_fn
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
            expand_attention_head_size=expand_attention_head_size
        )
        self.dropout = nn.Dropout(dropout)
        self.conv = ConvModule(self.embedding_dim, kernel_size=3, dropout_rate=dropout, bn=bn)
        self.feed_forward_out = FeedForward(self.embedding_dim, ffn_embedding_dim, activation_dropout)
        self.layer_norm_first = layer_norm_first

        assert self.layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
        pos_bias=None
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        x = x + 0.5 * self.feed_forward_in(x)

        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn, pos_bias = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
            position_bias=pos_bias
        )
        x = self.dropout(x)
        x = residual + x

        x = x + self.conv(x)

        x = x + 0.5 * self.feed_forward_out(x)
        
        x = self.final_layer_norm(x)

        return x, attn, pos_bias





