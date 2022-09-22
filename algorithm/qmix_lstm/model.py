import torch.nn as nn
import torch

class Model(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.linear1 = torch.nn.Linear(args.n_state, args.n_hidden)
        self.linear2 = torch.nn.Linear(args.n_hidden, args.n_rb)

        # store args
        self.args = args
        # initialize network
        self.lstm = nn.LSTM(args.n_state, args.n_hidden, args.n_layer, 
                           batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        self.fc   = nn.Linear(args.n_hidden * 2, args.n_rb)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.args.n_layer * 2, x.size(0), self.args.n_hidden, device=self.args.device)
        c0 = torch.zeros(self.args.n_layer * 2, x.size(0), self.args.n_hidden, device=self.args.device)
        # Forward propagate GRU
        x, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step and apply relu
        x = self.relu(x[:, -1, :])
        # Decode the action
        x = self.fc(x)
        return x
