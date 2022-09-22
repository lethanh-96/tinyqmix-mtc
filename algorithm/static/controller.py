import numpy as np
import torch

class Controller:

    def __init__(self, env, args):
        self.args = args

    def select_np_action(self, states):
        return np.array([i % self.args.n_rb for i in range(self.args.n_node)])
