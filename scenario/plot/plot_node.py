import matplotlib.pyplot as plt
import numpy as np
from . import util

def plot_node(args):
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
    n_nodes = [12, 24, 48, 96]
    metric = 'delay'

    # initialize plot
    r = np.arange(len(n_nodes))
    width = 0.13
    xticks  = r + (len(algorithms) - 1) / 2 * width
    tick_labels = n_nodes

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

    # initialize mean
    mean = {}
    std  = {}
    for algorithm in algorithms:
        mean[algorithm] = []
        std[algorithm]  = []

    for n_node in n_nodes:
        for algorithm in algorithms:
            # set
            args.algorithm = algorithm
            args.n_node    = n_node
            args.n_rb      = int(n_node / 6)
            # load
            df = util.load(args)
            # if 'centralize' in algorithm:
            #     y = df[metric].to_numpy()[:-240 * 6]
            #     y = y.reshape(240, 44)
            #     y_front = y[:, 0].reshape(-1, 1)
            #     y = np.concatenate([y_front, y_front, y_front, y_front, y_front, y_front, y], axis=1)
            #     y = y.reshape(-1)
            # else:
            y = df[metric].to_numpy()# [:2000]
            print(n_node, algorithm, len(y))
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
    plt.xlabel('Number of devices')
    # plt.legend()
    # plt.yscale('log')
    plt.savefig('figure/node_10.svg')
    plt.show()
