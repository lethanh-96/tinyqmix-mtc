from scipy.spatial.distance import hamming
import numpy as np
import torch

class RewardMonitor:

    def __init__(self, net, state_monitor):
        # save data
        self.net = net
        self.state_monitor = state_monitor
        # initialize
        self.reset()

    def reset(self):
        # initialize
        self.delay = np.zeros(self.net.args.n_node)
        self.success_rate = np.zeros(self.net.args.coherrent_time)
        self.compute_optimal_actions()

    def update(self, step, action, success, collision):
        args = self.net.args
        sr = self.state_monitor.success_rate
        self.delay += self.net.peak_delay
        if args.protocol == 'oma_rb':
            rb = action
            self.success_rate[step] = np.mean([sr[i, rb[i]] for i in range(args.n_node)])
        elif args.protocol == 'ofdma':
            self.success_rate[step] = np.mean([sr[i, :] for i in range(args.n_node)])
        else:
            raise NotImplementedError

    def compute_optimal_actions(self):
        # extract parameters
        args = self.net.args
        # extract sending rate
        node_indices  = np.array([i for i in range(args.n_node)])
        rate_per_node = self.state_monitor.traffic

        # initialize
        opt_action = np.zeros(args.n_node, dtype=np.int32)
        rate_per_rb = np.zeros(args.n_rb)
        # waterfilling
        while len(node_indices) > 0:
            # choose node with max rate
            # idx = self.random_argmax(rate_per_node)
            idx = np.argmax(rate_per_node)
            node_idx = node_indices[idx]
            # choose rb with min rate
            rb_idx = np.argmin(rate_per_rb)
            # rb_idx = self.random_argmin(rate_per_rb)
            # fill node to rb
            opt_action[node_idx] = rb_idx
            rate_per_rb[rb_idx] += rate_per_node[idx]
            # remove node indices
            node_indices  = np.delete(node_indices, idx)
            rate_per_node = np.delete(rate_per_node, idx)
        self.opt_action = opt_action

    def get(self, action):
        args = self.net.args
        sr = self.state_monitor.success_rate
        rb = action

        # reward formula -> edit this
        if args.reward == 'delay':
            reward = - np.sum(self.delay) / (5 * args.max_cw * args.coherrent_time)
        elif args.reward == 'hamming':
            reward = - hamming(action, self.opt_action)
        elif args.reward == 'sr':
            reward = np.mean(self.success_rate)
        elif args.reward == 'combined':
            reward = - np.sum(self.delay) / (5 * args.max_cw * args.coherrent_time)
            reward -= (1 - np.mean([sr[i, rb[i]] for i in range(args.n_node)]))
        return reward # break down component if needed here
