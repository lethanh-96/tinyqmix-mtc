import numpy as np

class StateMonitor:

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
        self.delay    = np.zeros([args.n_node, args.n_rb])
        self.success  = np.zeros(args.n_node)
        self.transmit = np.zeros(args.n_node)
        self.action   = np.zeros(args.n_node)

    def update(self, action, success, collision):
        # extract param
        net  = self.net
        args = net.args        
        beta = args.beta
        # update data
        self.traffic  = beta * self.traffic  + (1 - beta) * net.n_traffic
        peak_delay = net.peak_delay
        for i in range(net.args.n_node):
            rb = action[i]
            for j in range(net.args.n_rb):
                if j == rb:
                    self.delay[i, j] = beta * self.delay[i, j] + (1 - beta) * peak_delay[i]
                else:
                    self.delay[i, j] = beta * self.delay[i, j] + (1 - beta) * 0
        self.success  = beta * self.success  + (1 - beta) * success
        self.transmit = beta * self.transmit + (1 - beta) * net.n_transmit
        self.action   = action

    @property
    def success_rate(self):
        # extract data
        transmit = self.transmit
        success  = self.success
        # mask zeroes
        idx = np.where(transmit == 0)
        transmit[idx] = 1
        # compute rate
        r = self.success / self.transmit
        r[idx] = 1
        return r

    @property
    def traffic_estimation_mse(self):
        return np.mean((self.traffic - self.net.spatial_p * self.net.args.lamda) ** 2)

    @property
    def success_estimation_mse(self):
        return np.mean((self.success - self.net.spatial_p * self.net.args.lamda) ** 2)

    def get(self):
        return np.stack([self.traffic, self.delay[:, 0], self.delay[:, 1], self.action]).T