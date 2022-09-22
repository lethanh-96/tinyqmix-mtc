import matplotlib.pyplot as plt
import numpy as np
from . import util

def plot_dynamic(args):
    algorithms = [
        'dql', 
        'random', 
        'rr',
        'centralize_fast', 
        'qmix_lstm', 
        'qmix', 
        'optimal',
    ]
    labels = [
        'DQL',
        'Random',
        'RR',
        'WF',
        'QLBT',
        'QMIX',
        'WFLB'
    ]
    metric = 'delay'
    n_steps = [
        72000,# 300 / (0.5e-3 * 50),
        60 / (0.5e-3 * 50),
        10 / (0.5e-3 * 50),
    ]
    n_nodes = [12]

    # initialize plot
    r = np.arange(len(n_steps))
    width = 0.11
    xticks = r + (len(algorithms) - 1) / 2 * width
    tick_labels = ['Static', 'Dynamic (60s)', 'Dynamic (10s)']
    # tick_labels = ['Dynamic (60s)', 'Dynamic (10s)']


    # plot design
    fcs = [
        'white',
        'white',
        'gray',
        'gray',
        'green',
        'red',
        'blue',
    ]
    ecs = [
        'black',
        'black',
        'black',
        'black',
        'green',
        'red',
        'blue',
    ]
    hatches = [
        '',
        '///',
        '',
        '///',
        '',
        '',
        '',
    ]


    for n_node in n_nodes:
        # initialize mean
        mean = {}
        std  = {}
        for algorithm in algorithms:
            mean[algorithm] = []
            std[algorithm]  = []

        # gather mean
        for n_step in n_steps:
            for algorithm in algorithms:
                # set
                args.algorithm = algorithm
                args.n_step    = n_step
                args.n_node    = n_node
                args.n_rb      = int(args.n_node / 6)
                # load
                df = util.load(args)
                # if 'centralize' in algorithm:
                #     y = df[metric].to_numpy()[:-240 * 6]
                #     y = y.reshape(240, 44)
                #     y_front = y[:, 0].reshape(-1, 1)
                #     y = np.concatenate([y_front, y_front, y_front, y_front, y_front, y_front, y], axis=1)
                #     y = y.reshape(-1)
                # else:
                y = df[metric].to_numpy()
                # y = y[int(len(y) // 2):]
                # y = y[2000:]
                print(n_step, n_node, algorithm, len(y))
                mean[algorithm].append(np.mean(y))
                std[algorithm].append(np.std(y))

        # plot
        plt.figure(figsize=(8, 4))
        for i, algorithm in enumerate(algorithms):
            plt.bar(r + width * i, mean[algorithm], 
                    label=labels[i], 
                    width=width,
                    fc=fcs[i],
                    ec=ecs[i],
                    hatch=hatches[i],
            )

            # plotline, caps, barlinecols =\
            # plt.errorbar(r + width * i, mean[algorithm], 
            #              yerr=std[algorithm],
            #              xlolims=mean[algorithm],
            #              lolims=True,
            #              linestyle='none',
            #              ecolor='black',
            # )

        plt.xticks(ticks=xticks, labels=tick_labels)
        plt.ylabel('Average delay (ms)')
        # plt.xlabel('Time (minutes)')
        # plt.legend()
        plt.savefig(f'figure/dynamic_{args.n_node}_{args.n_rb}.svg')
        # plt.title(f'figure/dynamic_{args.n_node}_{args.n_rb}.svg')
        # plt.ylim((-1, 150))
        # plt.yscale('log')
        plt.show()
        break
