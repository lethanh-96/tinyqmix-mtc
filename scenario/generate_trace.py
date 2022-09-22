import numpy as np
import time
import os

def create_trace_path(args):
    if not os.path.exists(args.trace_dir):
        os.makedirs(args.trace_dir)
    return f'{args.trace_dir}/{args.traffic},{args.spatial}_{args.n_node}_{args.n_step}.npz'

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

def generate_trace(args):
    args.n_step = args.n_test_step
    # initialize trace
    T = args.n_step * args.coherrent_time * args.n_test_episode
    T_episode  = args.n_step * args.coherrent_time
    N = args.n_node
    trace = np.zeros([T, N], dtype=np.uint8)
    spatial_ps = []

    for t in range(0, T):
        # compute spatial pdf
        if t % T_episode == 0:
            spatial_p = create_spatial_p(args)
            spatial_ps.append(spatial_p)
        trace[t, :] = np.random.poisson(lam=args.lamda * spatial_p)
    # save
    print('[+] saving')
    path = create_trace_path(args)
    tic = time.time()
    data = {
        'spatial_ps' : np.array(spatial_ps),
        'trace': trace,
    }
    np.savez_compressed(path, **data)
    print(f'    {time.time() - tic:0.1}(s)')