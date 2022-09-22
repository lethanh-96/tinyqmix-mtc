import numpy as np
import torch

class Controller:

    def __init__(self, env, args):
        self.args = args
        self.env  = env
        self.action = None

    def __select_np_action(self, states):
        # extract parameters
        args = self.args
        env  = self.env
        # extract sending rate
        r = np.arange(args.coherrent_time * args.n_rb) + 1 
        r = r % args.n_node
        r = r.reshape(args.coherrent_time, args.n_rb)
        schedules = np.zeros([args.n_node, args.coherrent_time, args.n_rb])
        for i in range(args.n_node):
            schedules[i] = (r == i+1).astype(np.int32)
        return schedules

    def select_np_action(self, states):
        if self.action is None or self.env.now % 2000 == 0: # update every second, not coherrent time
            self.action = self.__select_np_action(states)
        return self.action
