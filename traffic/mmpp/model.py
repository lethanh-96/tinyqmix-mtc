import numpy as np

class Model:

    def __init__(self, spatial_p_, lambda_):
        self.spatial_p_ = spatial_p_
        self.lambda_    = lambda_

    def n_packet(self, t):
        return np.random.poisson(self.lambda_ * self.spatial_p_)

    def spatial_p(self, t):
        return self.spatial_p_