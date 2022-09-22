import numpy as np

class Controller:

    def __init__(self, env, args):
        self.args = args
        self.env  = env

    def select_np_action(self, states):
        # extract parameters
        args = self.args
        env  = self.env
        # extract sending rate
        node_indices  = np.array([i for i in range(args.n_node)])
        rate_per_node = np.array([node.traffic_model.spatial_p(env.now) for node in env.net.nodes])
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
        