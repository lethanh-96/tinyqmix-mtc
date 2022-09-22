import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from . import util

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_ma_static(args):
    algorithms = [
        'dql',
        # 'random',
        # 'rr',
        # 'centralize_fast', 
        'qmix', 
        'optimal',
    ]

    labels = [
        'DDQL',
        # 'Random',
        # 'RR',
        # 'WF',
        'QMIX',
        'WFLB'
    ]

    w           = 40
    n_node      = 12
    metric      = 'delay'
    colors      = ['cyan', 
                   # 'dimgray', 
                   # 'orange', 
                   # 'limegreen', 
                   'red', 
                   'blue'
                   ]
    tick_labels = np.arange(6)
    xticks      = tick_labels * 2400
    n_step      = 12000

    for j, algorithm in enumerate(algorithms):
        # set 
        args.algorithm = algorithm
        args.n_node    = n_node
        args.n_rb      = int(n_node / 6)
        args.n_step    = n_step

        # load
        df = util.load(args)
        y = df[metric].to_numpy()
        y = moving_average(y, w)

        # plot
        plt.plot(y,
                 color=colors[j], 
                 label=labels[j], 
                 markersize=10)
    plt.ylabel('Average delay (ms)')
    plt.xlabel('Time (minutes)')
    plt.xticks(ticks=xticks, labels=tick_labels)
    plt.legend()
    plt.savefig('figure/mean_time_12_2_300.svg')
    plt.show()
