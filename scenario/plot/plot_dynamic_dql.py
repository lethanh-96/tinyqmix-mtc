import matplotlib.pyplot as plt
import numpy as np
from . import util

def plot_dynamic_dql(args):
    algorithms = [
        'dql',
        'random', 
        'static',
        'centralize', 
        'centralize_fast', 
        'qmix_lstm', 
        'qmix', 
        'optimal',
    ]
    labels = [
        'DQL',
        'Random',
        'RR',
        'WF (1s)',
        'WF (25ms)',
        'QLBT',
        'QMIX',
        'WF (known traffic distribution)'
    ]
    metric = 'delay'
    n_steps = [
        300 / (0.5e-3 * 50),
        60 / (0.5e-3 * 50),
        10 / (0.5e-3 * 50),
    ]
    n_nodes = [12, 24]

    # initialize plot
    r = np.arange(len(n_steps))
    width = 0.11
    xticks = r + (len(algorithms) - 1) / 2 * width
    tick_labels = ['Static', 'Dynamic (60s)', 'Dynamic (10s)']

    # plot design
    fcs = [
        'white',
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
        'black',
        'breen',
        'red',
        'blue',
    ]
    hatches = [
        '///',
        '',
        '++',
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
                y = df[metric].to_numpy()
                # if 'centralize' in algorithm:
                #     y = df[metric].to_numpy()[:-240 * 6]
                #     y = y.reshape(240, 44)
                #     y_front = y[:, 0].reshape(-1, 1)
                #     y = np.concatenate([y_front, y_front, y_front, y_front, y_front, y_front, y], axis=1)
                #     y = y.reshape(-1)
                # else:
                #     y = df[metric].to_numpy()
                # y = y[int(len(y) // 2):]
                # y = y[2000:]
                mean[algorithm].append(np.mean(y))
                std[algorithm].append(np.std(y))

        # plot
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
        plt.xlabel('Time (minutes)')
        # plt.legend()
        plt.savefig(f'figure/dynamic_{args.n_node}_{args.n_rb}.svg')
        plt.title(f'figure/dynamic_{args.n_node}_{args.n_rb}.svg')
        plt.show()
        # plt.cla()
