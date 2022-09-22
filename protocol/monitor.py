from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np
import tqdm
import os

class Monitor:

    def __init__(self, env, args):
        # store access point and name, env, args, etc
        self.env     = env
        self.args    = args
        # initialize progress bar
        if args.mode == 'train':
            self.bar = tqdm.tqdm(range(int(self.args.n_step * self.args.n_train_episode)))
        else:
            self.bar = tqdm.tqdm(range(int(self.args.n_step * self.args.n_test_episode)))
        # initialize writer
        self.__create_tensorboard_writer()
        self.__create_csv_writer()

    # =====================================================================================
    # I/O management
    # =====================================================================================
    @property
    def label(self):
        args = self.args
        label  = f'{args.algorithm},{args.state},{args.reward}_{args.spatial}'
        label += f'_{args.n_rb},{args.n_node}_{int(args.n_step/40)}'
        return label

    def __create_tensorboard_writer(self):
        # extract parameters
        args = self.args
        # make directory if needed
        folder = f'{args.tensorboard_dir}/{args.mode}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        # construct folder name
        folder = f'{folder}/{self.label}'
        # remove old folder
        os.system(f'rm -rf {folder}')
        # create the tensorboard writer
        self.tensorboard_writer = SummaryWriter(folder)

    def __create_csv_writer(self):
        # extract parameters
        args = self.args
        # make directory if needed
        folder = f'{args.csv_dir}/{args.mode}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        # construct csv_path
        path = f'{folder}/{self.label}.csv'
        # remove old path
        os.system(f'rm -rf {path}')
        # create csv writer
        self.csv_writer = open(path, 'a+')

    def __del__(self):
        self.csv_writer.close()

    def __update_time(self):
        self.bar.update(1)

    def __update_description(self, **kwargs):
        display_kwargs = {}
        display_kwargs['now'] = self.env.now
        display_kwargs.update(**kwargs)
        self.bar.set_postfix(**display_kwargs)

    def __display(self):
        self.bar.display()

    def __update_tensorboard(self, stats):
        for key in stats.keys():
            value = stats[key]
            self.tensorboard_writer.add_scalar(key, value, global_step=self.env.now)
    # =====================================================================================

    # =====================================================================================
    # DATA management
    # =====================================================================================
    def compute_fairness(self, info):
        transmit = info['transmit']
        success  = info['success']
        rate     = np.array([s/t if t > 0 else 1 for s, t in zip(info['success'], info['transmit'])])
        if np.sum(rate ** 2) == 0:
            fairness = 1
        else:
            fairness = np.sum(rate) ** 2 / len(rate) / np.sum(rate ** 2)
        return fairness

    def __update_csv(self, stats):
        line  = f'{self.env.now:0.1f}'
        line += f',"{stats["n/transmit (pkt/slot/node)"]}"'
        line += f',"{stats["n/success (pkt/slot/node)"]}"'
        line += f',"{stats["n/collision (pkt/slot/node)"]}"'
        line += f',"{stats["n/traffic (pkt/slot/node)"]}"'
        line += f',"{stats["n/drop (pkt/slot/node)"]}"'
        line += f',"{stats["n/queue_length (pkt/slot/node)"]}"'
        line += f',"{stats["avg/delay (ms)"]}"'
        line += f',"{stats["avg/good_put"]}"'
        line += f',"{stats["avg/fairness"]}"'
        if 'loss' in stats.keys():
            line += f',"{stats["rl/loss"]}"'
        if 'eps' in stats.keys():
            line += f',"{stats["rl/eps"]}"'
        line += '\n'
        self.csv_writer.write(line)

    def get_stats(self, info): # TODO ADD PEAK DELAY
        # extract statistics
        stats = {
            f'n/transmit (pkt/slot/node)'     : np.mean(info['transmit']) / self.args.coherrent_time,
            f'n/success (pkt/slot/node)'      : np.mean(info['success']) / self.args.coherrent_time,
            f'n/collision (pkt/slot/node)'    : np.mean(info['collision']) / self.args.coherrent_time,
            f'n/traffic (pkt/slot/node)'      : np.mean(info['traffic']) / self.args.coherrent_time,
            f'n/drop (pkt/slot/node)'         : np.mean(info['drop']) / self.args.coherrent_time,
            f'n/queue_length (pkt/slot/node)' : np.mean(info['queue_length']) / self.args.coherrent_time,
            f'n/cw (slot)'                    : np.mean(info['cw']) / self.args.coherrent_time,
            f'avg/traffic_mse'                : np.mean(info['cw']) / self.args.coherrent_time,
            f'avg/success_mse'                : np.mean(info['cw']) / self.args.coherrent_time,
            f'avg/delay (ms)'                 : np.mean(info['delay']) * self.args.slot_duration,
            f'avg/good_put'                   : np.mean(info['success']) / self.args.coherrent_time * 100,
            f'avg/fairness'                   : self.compute_fairness(info),
            f'rl/reward'                      : info['reward'],
            f'state/traffic'                  : info['state/traffic'],
            f'state/sr0'                      : info['state/sr0'],
            f'state/sr1'                      : info['state/sr1'],
            f'state/rbs'                      : info['state/rbs'],
            f'state/traffic_p'                : info['state/traffic_p'],
        }
        if 'loss' in info.keys():
            stats[f'rl/loss'] = info['loss']
        if 'eps' in info.keys():
            stats['rl/eps'] = np.sum(info['eps'])
        if 'memory_size' in info.keys():
            stats['n/memory_size'] = info['memory_size']
        return stats

    def get_pbar_stats(self, stats):
        pbar_stats = {}
        if 'n/memory_size' in stats:
            pbar_stats['memory'] = stats['n/memory_size']
        return pbar_stats
    # =====================================================================================

    # =====================================================================================
    # API called by RL algorithm
    # =====================================================================================
    def step(self, info):
        # extract stats from all stations
        stats = self.get_stats(info)
        # update progress bar
        self.__update_time()
        self.__update_description(**self.get_pbar_stats(stats))
        self.__display()
        # log to tensorboard
        self.__update_tensorboard(stats)
        # log to csv
        self.__update_csv(stats)

