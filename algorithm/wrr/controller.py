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
        node_indices  = np.array([i for i in range(args.n_node)])
        rate_per_node = np.sum(env.state_monitor.success, axis=1)
        sum_p = np.sum(rate_per_node)
        if sum_p > 0:
            p = rate_per_node / sum_p
        else:
            p = np.full([args.n_node], 1. / args.n_node)
        # generate ofdma allocation for \uptau x M slots
        schedules = np.zeros([args.n_node, args.coherrent_time, args.n_rb])
        r = np.random.choice(np.arange(args.n_node) + 1, 
                             p=p, 
                             size=[args.coherrent_time, args.n_rb])
        for i in range(args.n_node):
            schedules[i] = (r == i+1).astype(np.int32)
        return schedules

    def select_np_action(self, states):
        if self.action is None or self.env.now % 2000 == 0: # update every second, not coherrent time
            self.action = self.__select_np_action(states)
        return self.action
