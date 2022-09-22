import torch
from pthflops import count_ops

class Model(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.rand(args.n_state))
        self.var  = torch.nn.Parameter(torch.rand(args.n_state))
        self.device  = args.device

    def forward(self, x):
        q = (x - self.mean) / self.var
        return q

def flops_preprocessor(args):
    for n_node in [12, 24, 48, 96]:
        args.n_node = n_node
        args.n_rb = int(args.n_node / 6)
        args.n_state = args.n_rb + 2

        model = Model(args).to(args.device)
        x = torch.rand(args.n_state).to(args.device)
        count_ops(model, x)
        print(args.n_state * 2)
        # 8 12 20 36