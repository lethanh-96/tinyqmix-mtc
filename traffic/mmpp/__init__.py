import numpy as np

from .model import Model

def create_spatial_p(args):
    if args.spatial == 'highlow':
        spatial_pdf = []
        for i in range(args.n_node):
            if np.random.rand() < 1 / 5:
                spatial_pdf.append(0.3 / 6)
            else:
                spatial_pdf.append(0.025 / 6)
        return np.array(spatial_pdf)
    elif args.spatial == 'uniform':
        return np.random.rand(args.n_node)
    else:
        raise NotImplementedError

def set_traffic_models(net, args):
    # create spatial pdf
    spatial_p = create_spatial_p(args)
    # assign to each node
    for i, node in enumerate(net.nodes):
        node.traffic_model = Model(spatial_p[i], args.lamda)
