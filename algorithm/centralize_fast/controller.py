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
        # initialize
        action = np.ones(args.n_node, dtype=np.int32)
        rate_per_rb = np.zeros(self.args.n_rb)
        # waterfilling
        while len(node_indices) > 0:
            # choose node with max rate
            idx = np.argmax(rate_per_node)
            node_idx = node_indices[idx]
            # choose rb with min rate
            rb_idx = np.argmin(rate_per_rb)
            # fill node to rb
            action[node_idx]     = rb_idx
            rate_per_rb[rb_idx] += rate_per_node[idx]
            # remove node indices
            node_indices  = np.delete(node_indices, idx)
            rate_per_node = np.delete(rate_per_node, idx)
        return action
        
    def select_np_action(self, states):
        self.action = self.__select_np_action(states)
        self.env.initialize_info()
        self.env.reward_monitor.reset()
        for step in range(6):
            # generate traffic and skip 1 step
            self.env.net.generate_traffic()
            # transmit packet
            for node in self.env.net.nodes:
                node.is_transmitting = False
            # compute info
            success, collision = self.env.net.compute_info(self.action)
            # store info
            self.env.update_info(success, collision)
            # compute reward
            self.env.reward_monitor.update(step, self.action, success, collision)
            self.env.state_monitor.update(self.action, success, collision)
            # acknowledgement
            self.env.net.ack(collision)
            self.env.net.now += 1
        return self.action
