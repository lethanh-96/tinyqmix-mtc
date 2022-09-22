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
        self.success  = np.zeros([args.n_node, args.n_rb])
        self.transmit = np.zeros([args.n_node, args.n_rb])
        self.busy     = np.zeros([args.n_node, args.n_rb])
        self.action   = np.zeros(args.n_node)

    def update(self, action, success, collision):
        # extract param
        net  = self.net
        args = net.args        
        beta = args.beta
        # update traffic
        self.traffic  = beta * self.traffic  + (1 - beta) * net.n_traffic
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
                    else:
                        self.transmit[i, j] = beta * self.transmit[i, j] + (1 - beta) * 0
                        self.success[i, j]  = beta * self.success[i, j]  + (1 - beta) * 0
                    self.busy[i, j] = beta * self.busy[i, j] + (1 - beta) * busy[j]
        else:
            pass
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
        r = success / transmit
        r[idx] = 0.5
        return r

    @property
    def traffic_estimation_mse(self):
        return np.mean((self.traffic - self.net.spatial_p * self.net.args.lamda) ** 2)

    @property
    def success_estimation_mse(self):
        return np.mean((np.sum(self.success, axis=1) - self.net.spatial_p * self.net.args.lamda) ** 2)

    def get(self):
        if self.net.args.protocol == 'oma_rb':
            if self.net.args.state == 'local':
                success_rate = self.success_rate
                states = [self.traffic, self.action] +\
                         [success_rate[:, i] for i in range(self.net.args.n_rb)]
            elif self.net.args.state == 'ack':
                busy   = self.busy
                states = [self.traffic, self.action] +\
                         [busy[:, i] for i in range(self.net.args.n_rb)]
            return np.stack(states).T
        else:
            return np.expand_dims(np.zeros([self.net.args.n_rb + 2]), axis=0).T # dummy state