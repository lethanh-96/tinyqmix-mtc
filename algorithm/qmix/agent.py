import numpy as np
import torch
import math

from .model import Model

class Agent:

    def __init__(self, env, args):
        self.args = args
        self.env  = env
        self.__init_policy()

    def __init_policy(self):
        self.policy_net = Model(self.args).to(self.args.device)
        self.target_net = Model(self.args).to(self.args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
