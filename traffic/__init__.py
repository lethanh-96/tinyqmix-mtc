from . import mmpp
from . import trace

def set_traffic_models(net, args):
    if args.mode == 'train':
        mmpp.set_traffic_models(net, args)
    elif args.mode == 'test':
        trace.set_traffic_models(net, args)
    else:
        raise NotImplementedError
