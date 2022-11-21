from . import centralize_fast
from . import centralize
from . import optimal
from . import random
from . import static
from . import tinyqmix
from . import qmix_lstm
from . import dql
from . import wrr
from . import rr

def create_controller(env, args):
    if args.algorithm == 'centralize':
        return centralize.Controller(env, args)
    elif args.algorithm == 'centralize_fast':
        return centralize_fast.Controller(env, args)
    elif args.algorithm == 'random':
        return random.Controller(env, args)
    elif args.algorithm == 'static':
        return static.Controller(env, args)
    elif args.algorithm == 'optimal':
        return optimal.Controller(env, args)
    elif args.algorithm == 'tinyqmix':
        return tinyqmix.Controller(env, args)
    elif args.algorithm == 'qmix_lstm':
        return qmix_lstm.Controller(env, args)
    elif args.algorithm == 'dql':
        return dql.Controller(env, args)
    elif args.algorithm == 'wrr':
        return wrr.Controller(env, args)
    elif args.algorithm == 'rr':
        return rr.Controller(env, args)
    else:
        raise NotImplementedError
