import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

cuda = torch.cuda.is_available()
DEVICE = "cuda" if cuda else "cpu"
if cuda:
    print("CUDA GPU!")
else:
    print("CPU!")


def w(v):
    if cuda:
        return v.cuda()
    return v


def detach_var(v):
    var = Variable(v.data, requires_grad=True)
    var.retain_grad()
    return var


class LearnedOptimizer(nn.Module):
    def __init__(self, device, preproc=False, hidden_sz=20, preproc_factor=10.0,
                 max_steps=10, meta_lr=1e-3):
        super().__init__()
        self.device = device
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
        self.hidden_states = None
        self.cell_states = None

        # Optimization Info
        self.max_steps = max_steps
        self.loss_to_optimize = 0.0
        self.num_steps = 0.0

        # "Meta Optimizer" to optimize this learned optimizer's own weights
        self.meta_opt = torch.optim.Adam(self.parameters(), lr=meta_lr)

    def forward(self, inp, hidden, cell, n_params):
        if self.hidden_states is None:
            self.hidden_states = [
                w(Variable(torch.zeros(n_params, self.hidden_sz))) for _ in range(2)]
            self.cell_states = [w(Variable(torch.zeros(n_params, self.hidden_sz)))
                                for _ in range(2)]

        if self.preproc:
            # Implement preproc described in Appendix A

            # Note: we do all this work on tensors, which means
            # the gradients won't propagate through inp. This
            # should be ok because the algorithm involves
            # making sure that inp is already detached.
            inp = inp.data
            inp2 = torch.zeros(inp.size()[0], 2, device=self.device)
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (
                torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()

            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (
                float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = Variable(inp2)
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)

    def update(self, new_loss):
        self.loss_to_optimize = self.loss_to_optimize + new_loss
        self.num_steps += 1

        if self.num_steps > self.max_steps:
            if self.train:
                self.meta_opt.zero_grad()
                self.loss_to_optimize.backward()
                self.meta_opt.step()

            # reset computation graph
            self.loss_to_optimize = 0.0
            torch.cuda.empty_cache()

    def reset_grad(self, reset_history=False):
        self.grad_len = 0
        if reset_history:
            self.feat_adaptor_history = None
            self.error_encoder_history = None
        else:
            self.feat_adaptor_history = [detach_var(v).to(
                self.device) for v in self.feat_adaptor_history]
            self.error_encoder_history = [detach_var(v).to(
                self.device) for v in self.error_encoder_history]
