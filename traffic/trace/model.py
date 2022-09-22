import numpy as np

class Model:

    def __init__(self, trace, spatial_ps, args):
        self.trace      = trace
        self.spatial_ps = spatial_ps
        self.T          = args.n_step * args.coherrent_time * args.n_test_episode
        self.T_episode  = args.n_step * args.coherrent_time

    def spatial_p(self, t):
        return self.spatial_ps[int(np.floor((t % self.T) / self.T_episode))]

    def n_packet(self, t):
        return self.trace[t % self.T]