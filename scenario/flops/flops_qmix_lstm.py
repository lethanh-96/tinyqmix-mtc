from pthflops import count_ops
import torch.nn as nn
import torch

# class Model(torch.nn.Module):

#     def __init__(self, args):
#         super().__init__()
#         # store args
#         self.args = args
#         # initialize network
#         self.lstm = nn.LSTM(args.n_state, args.n_hidden, args.n_layer, 
#                            batch_first=True, bidirectional=True)
#         self.fc   = nn.Linear(args.n_hidden * 2, args.n_rb)

#     def forward(self, x):
#         # Set initial states
#         # Forward propagate GRU
#         out, _ = self.lstm(x, (self.h0, self.c0))
#         # Decode the hidden state of the last time step
#         out = self.fc(out[:, -1, :])
#         return out

class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lstm_in     = torch.nn.Linear(args.n_state, args.n_hidden)
        self.lstm_hidden = torch.nn.Linear(args.n_hidden, args.n_hidden)
        self.fc = torch.nn.Linear(args.n_hidden, args.n_rb)
        batch_size = 1
        x = torch.rand(batch_size, args.sequence_length, args.n_state).to(args.device)
        self.h0 = torch.rand(args.n_hidden).to(args.device)
        self.c0 = torch.rand(args.n_state).to(args.device)

    def forward(self, x):
        for i in range(self.args.sequence_length):
            out = self.lstm_in(self.c0) + self.lstm_hidden(self.h0)
        out = self.fc(out)
        return out

def flops_qmix_lstm(args):
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
        batch_size = 1
        x = torch.rand(batch_size, args.sequence_length, args.n_state).to(args.device)
        count_ops(model, x)
        # 1298, 1476, 4936, 17808