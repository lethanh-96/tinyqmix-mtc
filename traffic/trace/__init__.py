import numpy as np
import os

from .model import Model

def create_trace_path(args):
    if not os.path.exists(args.trace_dir):
        os.makedirs(args.trace_dir)
    return f'{args.trace_dir}/{args.traffic},{args.spatial}_{args.n_node}_{args.n_step}.npz'

def set_traffic_models(net, args):
    # load trace for all nodes
    path  = create_trace_path(args)
    data  = np.load(path)
    trace = data['trace']
    spatial_ps = data['spatial_ps']
    # assign to each node
    for i, node in enumerate(net.nodes):
        node.traffic_model = Model(trace[:, i], spatial_ps[:, i], args)
