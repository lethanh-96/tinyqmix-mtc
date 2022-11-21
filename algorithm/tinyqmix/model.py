import torch

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