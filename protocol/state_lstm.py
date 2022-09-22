import numpy as np

class StateMonitorLstm:

    def __init__(self, net):
        # save data
        self.net = net
        # initialize
        self.reset()

    def reset(self):
        # extract param
        args = self.net.args
        # initialize data
        self.traffic  = np.zeros(args.n_node)
        self.success  = np.zeros([args.n_node, args.n_rb])
        self.transmit = np.zeros([args.n_node, args.n_rb])
        self.busy     = np.zeros([args.n_node, args.n_rb])
        self.action   = np.zeros(args.n_node)
        # initialize sequence data
        self.traffic_sequence  = [[0 for _ in range(args.sequence_length * args.coherrent_time)] for _ in range(args.n_node)]
        self.success_sequence  = [[0 for _ in range(args.sequence_length * args.coherrent_time)] for _ in range(args.n_node)]
        self.transmit_sequence = [[0 for _ in range(args.sequence_length * args.coherrent_time)] for _ in range(args.n_node)]
        self.action_sequence   = [[0 for _ in range(args.sequence_length * args.coherrent_time)] for _ in range(args.n_node)]

    def update(self, action, success, collision):
        # extract param
        net  = self.net
        args = net.args        
        beta = args.beta
        L    = args.sequence_length * args.coherrent_time
        # update traffic
        self.traffic  = beta * self.traffic  + (1 - beta) * net.n_traffic
        for i in range(args.n_node):
            self.traffic_sequence[i].append(net.n_traffic[i])
            while len(self.traffic_sequence[i]) > L:
                self.traffic_sequence[i].pop(0)
        # update transmit and success
        transmit = net.n_transmit
        busy     = net.busy
        if args.protocol == 'oma_rb':
            for i in range(net.args.n_node):
                rb = action[i]
                for j in range(net.args.n_rb):
                    if j == rb:
                        self.transmit[i, j] = beta * self.transmit[i, j] + (1 - beta) * transmit[i]
                        self.success[i, j]  = beta * self.success[i, j]  + (1 - beta) * success[i]
                        self.transmit_sequence[i].append(transmit[i])
                        while len(self.transmit_sequence[i]) > L:
                            self.transmit_sequence[i].pop(0)
                        self.success_sequence[i].append(success[i])
                        while len(self.success_sequence[i]) > L:
                            self.success_sequence[i].pop(0)
                    else:
                        self.transmit[i, j] = beta * self.transmit[i, j] + (1 - beta) * 0
                        self.success[i, j]  = beta * self.success[i, j]  + (1 - beta) * 0
        else:
            pass
        self.action = action
        for i in range(args.n_node):
            self.action_sequence[i].append(action[i])
            while len(self.action_sequence[i]) > L:
                self.action_sequence[i].pop(0)

    @property
    def traffic_estimation_mse(self):
        return np.mean((self.traffic - self.net.spatial_p * self.net.args.lamda) ** 2)

    @property
    def success_estimation_mse(self):
        return np.mean((np.sum(self.success, axis=1) - self.net.spatial_p * self.net.args.lamda) ** 2)

    def get_feature(self, x):
        args = self.net.args
        x = np.array(x)
        x = x.reshape(-1, args.sequence_length, args.coherrent_time)
        x = np.mean(x, axis=2)
        return x

    def get(self):
        states = [
            self.get_feature(self.traffic_sequence),
            self.get_feature(self.transmit_sequence),
            self.get_feature(self.success_sequence),
            self.get_feature(self.action_sequence),
        ]
        states = np.stack(states)
        states = np.transpose(states, axes=(1, 2, 0))
        return states

    @property
    def success_rate(self):
        # extract data
        transmit = self.transmit
        success  = self.success
        # mask zeroes
        idx = np.where(transmit == 0)
        transmit[idx] = 1
        # compute rate
        r = success / transmit
        r[idx] = 0.5
        return r