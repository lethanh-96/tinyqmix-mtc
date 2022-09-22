import torch 

class Preprocessor(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        # save args
        self.args = args
        # initialize
        self.n         = 0
        self.mean      = torch.zeros(args.n_state, device=args.device, dtype=torch.float32)
        self.mean_diff = torch.zeros(args.n_state, device=args.device, dtype=torch.float32)
        self.var       = torch.zeros(args.n_state, device=args.device, dtype=torch.float32)

    def observe(self, state):
        self.n         += 1
        last_mean       = self.mean.clone().detach()
        self.mean      += (state - self.mean) / self.n
        self.mean_diff += (state - last_mean) * (state - self.mean)
        self.var        = (self.mean_diff / self.n).clip(min=1e-2)

    def fit_transform(self, states):
        for state in states:
            self.observe(state)
        return (states - self.mean) / torch.sqrt(self.var)

    def transform(self, states):
        return (states - self.mean) / torch.sqrt(self.var)
