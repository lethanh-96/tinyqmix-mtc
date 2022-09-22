import numpy as np
import torch

class Controller:

    def __init__(self, env, args):
        self.args = args

    def select_np_action(self, state):
        return np.random.randint(low=0, 
                                 high=self.args.n_rb, 
                                 size=(self.args.n_node,))
