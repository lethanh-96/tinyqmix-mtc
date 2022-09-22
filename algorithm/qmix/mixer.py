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
        self.hyper_w1 = nn.Sequential(nn.Linear(n_input, n_hyper_hidden),
                                      nn.ReLU(),
                                      nn.Linear(n_hyper_hidden, n_embed * n_agent))
        self.hyper_b1 = nn.Linear(n_input, n_embed)
        self.hyper_w2 = nn.Sequential(nn.Linear(n_input, n_hyper_hidden),
                                      nn.ReLU(),
                                      nn.Linear(n_hyper_hidden, n_embed))
        self.V = nn.Sequential(nn.Linear(n_input, n_embed),
                               nn.ReLU(),
                               nn.Linear(n_embed, 1))

    def compute_policy_qs(self, states):
        if len(states.shape) == 2:
            qs = torch.stack([self.agents[i].policy_net(states[i]) for i in range(self.args.n_node)])\
                      .squeeze()
        elif len(states.shape) == 3:
            qs = []
            for i in range(self.args.n_node):
                q = self.agents[i].policy_net(states[:, i, :])
                # if np.random.rand() < self.args.dropout: # dropout 0: no dropout
                #     q = q.detach()
                qs.append(q)
            qs = torch.stack(qs).squeeze()
            qs = torch.transpose(qs, 0, 1)
        return qs

    def compute_target_qs(self, states):
        if len(states.shape) == 2:
            qs = torch.stack([self.agents[i].target_net(states[i]) for i in range(self.args.n_node)])\
                      .squeeze()
        elif len(states.shape) == 3:
            qs = torch.stack([self.agents[i].target_net(states[:, i, :]) for i in range(self.args.n_node)])\
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
        states = states.reshape(bs, -1)
        # generate parameter for first layer
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, n_agent, n_embed)
        b1 = b1.view(-1, 1, n_embed)
        # generate parameter for second layer
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(-1, n_embed, 1)
        # generate state dependent bias
        v = self.V(states).view(-1, 1, 1)
        # embed the qs to embeding space
        hidden = F.elu(torch.bmm(qs, w1) + b1)
        # compute q_tot from embeded vector
        y = torch.bmm(hidden, w2) + v
        # reshape q_tot and return
        q_tot = y.view(bs,)
        return q_tot
        