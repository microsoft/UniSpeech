import os
import tempfile
import math
import torch
from torch.utils.cpp_extension import load
from torch.autograd import Function
from torch.nn import Module



# TODO: this will fail if dist is not initialized
# compiling cuda loss on the fly. This assumes that we are running main
# script from inside main directory, i.e. "python train.py ..." and not
# from any other folder
"""
if not torch.cuda.is_available():
    raise ValueError("RNNTLoss without CUDA has not been implemented yet")
build_dir = os.path.join(tempfile.gettempdir(),
                         'rnntLoss_cpp-rank-{}'.format(0))
os.makedirs(build_dir, exist_ok=True)
rnntLoss = load(
    name='rnntLoss_cpp',
    sources=[os.path.join(os.getcwd(), 'fairseq', 'criterions', 'rnnt', 'rnntLoss.cpp'),
             os.path.join(os.getcwd(), 'fairseq', 'criterions', 'rnnt', 'rnnt.cu')],
    build_directory=build_dir,
    verbose=True
)
"""

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, "gradients only computed for acts - please "\
        "mark other tensors as not requiring gradients"


class _RNNT(Function):
    """
    RNNT loss forward propagation
    """
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens, blank=0, loss_scale=1.0):
        if torch.cuda.is_available():
            costs = torch.zeros(acts.shape[0]).cuda()
        else:
            costs = torch.zeros(acts.shape[0])

        if torch.cuda.is_available():
            cum = torch.cat(
                (torch.zeros(1, dtype=torch.int32, device="cuda"),
                 torch.cumsum(label_lens[:-1], 0, dtype=torch.int32)),
                0,
            )
        else:
            cum = torch.cat(
                (torch.zeros(1, dtype=torch.int32),
                 torch.cumsum(label_lens[:-1], 0, dtype=torch.int32)),
                0,
            )
        acts = torch.nn.functional.log_softmax(acts, dim=-1)
        import rnntLoss
        rnntLoss.transducer(acts, labels, act_lens, label_lens, costs, cum, blank, loss_scale)
        ctx.save_for_backward(acts)

        return costs.sum()

    """
    RNNT loss backward function to receive returned gradients
    """
    @staticmethod
    def backward(ctx, grad_output):
        grads = ctx.saved_tensors[0]
        return grads, None, None, None, None, None


class RNNTLoss(Module):
    """
    RNNT loss wrapper
    Parameters:
        acts: logit outputs
        labels: setenece pieces
        acts_lens: length of input logits
        label_lens: length of input setence peices
    """
    def __init__(self):
        super(RNNTLoss, self).__init__()
        self.rnnt = _RNNT.apply

    """
    RNNT loss wrapper forward function in which legal inputs are checked
    """
    def forward(self, acts, labels, act_lens, label_lens, blank=0, loss_scale=1.0):
        assert len(labels.size()) == 1
        _assert_no_grad(labels)
        _assert_no_grad(act_lens)
        _assert_no_grad(label_lens)

        return self.rnnt(acts, labels, act_lens, label_lens, blank, loss_scale)
