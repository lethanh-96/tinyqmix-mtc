import argparse
import torch

CHANNEL_COHERENCE = 50

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    # scenario
    parser.add_argument('--scenario', type=str, default='train')
    # algorithm
    parser.add_argument('--algorithm', type=str, default='qmix_lstm',
                        choices=['random', 'static', 'optimal', 'centralize', 'qmix', 'qmix_lstm', 
                                 'dql', 'centralize_fast', 'wrr', 'rr'])
    # general
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--n_step', type=int, default=int(10 / (0.5e-3 * CHANNEL_COHERENCE)))
    parser.add_argument('--n_train_step', type=int, default=int(10 / (0.5e-3 * CHANNEL_COHERENCE)))
    parser.add_argument('--n_test_step', type=int, default=int(10 / (0.5e-3 * CHANNEL_COHERENCE)))
    parser.add_argument('--slot_duration', type=float, default=0.5, help='ms')
    parser.add_argument('--device', type=str, default='cuda')
    # monitor
    parser.add_argument('--trace_dir', type=str, default='data/trace')
    parser.add_argument('--tensorboard_dir', type=str, default='data/tensorboard')
    parser.add_argument('--csv_dir', type=str, default='data/csv')
    parser.add_argument('--model_dir', type=str, default='data/model')
    parser.add_argument('--save_model_iter', type=int, default=10)
    # network
    parser.add_argument('--n_node', type=int, default=12)
    parser.add_argument('--coherrent_time', type=int, default=CHANNEL_COHERENCE, help='timeslots')
    parser.add_argument('--queue_length', type=float, default=16)
    parser.add_argument('--max_cw', type=float, default=17)
    parser.add_argument('--max_retry', type=float, default=16)
    # mac protocol
    parser.add_argument('--protocol', type=str, default='oma_rb',
                        choices=['oma_rb', 'ofdma'])
    parser.add_argument('--n_rb', type=int, default=2)
    # traffic
    parser.add_argument('--traffic', type=str, default='mmpp')
    parser.add_argument('--lamda', type=float, default=2)
    parser.add_argument('--spatial', type=str, default='highlow',
                        choices=['highlow', 'uniform'])
    # mdp
    # parser.add_argument('--beta', type=float, default=0.5, help='state estimation discount')
    parser.add_argument('--beta', type=float, default=0.999, help='state estimation discount')
    parser.add_argument('--reward', type=str, default='sr',
                        choices=['hamming', 'delay', 'sr', 'combined'])
    parser.add_argument('--state', type=str, default='local',
                        choices=['local', 'ack'])
    # dqn
    parser.add_argument('--n_train_episode', type=int, default=10000)
    parser.add_argument('--n_test_episode', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    # parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--memory_size', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--eps_start', type=float, default=0.9)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay', type=int, default=500)
    parser.add_argument('--train_frequency', type=int, default=32)
    # dqn model
    parser.add_argument('--n_hidden', type=int, default=8)
    # lstm model
    parser.add_argument('--sequence_length', type=int, default=10)
    parser.add_argument('--n_layer', type=int, default=1)
    # qmix model
    parser.add_argument('--n_embed', type=int, default=32)
    parser.add_argument('--n_mixer_hidden', type=int, default=64)
    # parse args
    args = parser.parse_args()
    # additional args
    if args.scenario == 'train': 
        args.mode = 'train'
        args.dropout = 1 - args.n_rb / args.n_node
    else:
        args.mode = 'test'
        args.dropout = 0
        # args.n_step = int(20 / (0.5e-3 * CHANNEL_COHERENCE))
    if 'qmix' in args.algorithm:
        args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')
    if args.n_node == 24:
        args.n_hidden = 8
        args.n_mixer_hidden = 128
    if args.n_node == 48:
        args.n_hidden = 16
        args.n_mixer_hidden = 256
    if args.n_node == 96:
        args.n_hidden = 32
        args.n_mixer_hidden = 512
    if args.n_node >= 12:
        args.n_rb = int(args.n_node / 6)
    return args

def print_args(args):
    pass
