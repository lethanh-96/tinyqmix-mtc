import torch
from pthflops import count_ops

class Model(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.linear1 = torch.nn.Linear(args.n_state, args.n_hidden)
        self.linear2 = torch.nn.Linear(args.n_hidden, args.n_rb)
        self.device  = args.device

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        q = self.linear2(x)
        return q

def flops_qmix(args):
    for n_node in [12, 24, 48, 96]:
        args.n_node = n_node
        args.n_rb = int(args.n_node / 6)
        args.n_state = args.n_rb + 2
        if args.n_node == 24:
            args.n_hidden = 8
            args.n_mixer_hidden = 128
        if args.n_node == 48:
            args.n_hidden = 16
            args.n_mixer_hidden = 256
        if args.n_node == 96:
            args.n_hidden = 32
            args.n_mixer_hidden = 512

        model = Model(args).to(args.device)
        x = torch.rand(args.n_state).to(args.device)
        count_ops(model, x)
        # 58, 92, 312, 1136