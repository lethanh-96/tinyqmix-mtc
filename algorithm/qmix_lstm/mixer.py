import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

class Mixer(torch.nn.Module):

    def __init__(self, agents, args):
        super().__init__()
        # for gradient flowing to agent's dnn
        for i, agent in enumerate(agents):
            setattr(self, f'policy_net_{i}', agent.policy_net)
        # for inference
        self.agents = agents
        self.args = args
        # extract hyper parameter
        n_input        = args.n_state * args.n_node
        n_hyper_hidden = args.n_mixer_hidden
        n_embed        = args.n_embed
        n_agent        = args.n_node
        # mixer's dnn
        self.hyper_lstm_w1   = nn.LSTM(n_input, n_hyper_hidden, args.n_layer, 
                                       batch_first=True, bidirectional=True)
        self.hyper_linear_w1 = nn.Linear(n_hyper_hidden * 2, n_embed * n_agent)

        self.hyper_lstm_b1   = nn.LSTM(n_input, n_embed // 2, args.n_layer, 
                                       batch_first=True, bidirectional=True)

        self.hyper_lstm_w2   = nn.LSTM(n_input, n_hyper_hidden, args.n_layer, 
                                       batch_first=True, bidirectional=True)
        self.hyper_linear_w2 = nn.Linear(n_hyper_hidden * 2, n_embed)

        self.hyper_lstm_V   = nn.LSTM(n_input, n_embed, args.n_layer, 
                                       batch_first=True, bidirectional=True)
        self.hyper_linear_V = nn.Linear(n_embed * 2, 1)

    def compute_policy_qs(self, states):
        if len(states.shape) == 3:
            qs = torch.stack([self.agents[i].policy_net(states[i].unsqueeze(0)) for i in range(self.args.n_node)]).squeeze()
        elif len(states.shape) == 4:
            qs = []
            for i in range(self.args.n_node):
                q = self.agents[i].policy_net(states[:, i, :, :])
                qs.append(q)
            qs = torch.stack(qs).squeeze()
            qs = torch.transpose(qs, 0, 1)
        return qs

    def compute_target_qs(self, states):
        if len(states.shape) == 3:
            qs = torch.stack([self.agents[i].target_net(states[i]) for i in range(self.args.n_node)])\
                      .squeeze()
        elif len(states.shape) == 4:
            qs = torch.stack([self.agents[i].target_net(states[:, i, :, :]) for i in range(self.args.n_node)])\
                      .squeeze()
            qs = torch.transpose(qs, 0, 1)
        return qs

    def compute_q_tot(self, qs, states):
        # extract hyper parameter
        bs      = qs.size(0)
        n_agent = self.args.n_node
        n_embed = self.args.n_embed

        # reshape states and qs
        qs = qs.view(bs, 1, n_agent)

        states = torch.transpose(states, 1, 2)
        states = states.reshape(bs, self.args.sequence_length, -1)

        # weight for first layer
        h0 = torch.zeros(self.args.n_layer * 2, states.size(0), self.args.n_mixer_hidden, device=self.args.device)
        c0 = torch.zeros(self.args.n_layer * 2, states.size(0), self.args.n_mixer_hidden, device=self.args.device)
        out, _ = self.hyper_lstm_w1(states, (h0, c0))
        out = self.hyper_linear_w1(out[:, -1, :])
        w1 = torch.abs(out)
        # bias for first layer
        h0 = torch.zeros(self.args.n_layer * 2, states.size(0), self.args.n_embed // 2, device=self.args.device)
        c0 = torch.zeros(self.args.n_layer * 2, states.size(0), self.args.n_embed // 2, device=self.args.device)
        out, _ = self.hyper_lstm_b1(states, (h0, c0))
        b1 = out[:, -1, :]
        # reshape first layer
        w1 = w1.view(-1, n_agent, n_embed)
        b1 = b1.view(-1, 1, n_embed)

        # weight for second layer
        h0 = torch.zeros(self.args.n_layer * 2, states.size(0), self.args.n_mixer_hidden, device=self.args.device)
        c0 = torch.zeros(self.args.n_layer * 2, states.size(0), self.args.n_mixer_hidden, device=self.args.device)
        out, _ = self.hyper_lstm_w2(states, (h0, c0))
        out = self.hyper_linear_w2(out[:, -1, :])
        w2 = torch.abs(out)
        w2 = w2.view(-1, n_embed, 1)

        # generate state dependent bias
        h0     = torch.zeros(self.args.n_layer * 2, states.size(0), self.args.n_embed, device=self.args.device)
        c0     = torch.zeros(self.args.n_layer * 2, states.size(0), self.args.n_embed, device=self.args.device)
        out, _ = self.hyper_lstm_V(states, (h0, c0))
        out    = self.hyper_linear_V(out[:, -1, :])
        v      = out.view(-1, 1, 1)

        # embed the qs to embeding space
        hidden = F.elu(torch.bmm(qs, w1) + b1)
        # compute q_tot from embeded vector
        y = torch.bmm(hidden, w2) + v
        # reshape q_tot and return
        q_tot = y.view(bs,)
        return q_tot
