import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from . import util

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_ma(args):
    algorithms = [
        'random',
        'rr',
        'centralize_fast', 
        'qmix', 
        'optimal',
    ]

    labels = [
        'Random',
        'RR',
        'WF',
        'TinyQMIX',
        'WFLB'
    ]

    w           = 40
    n_node      = 12
    metric      = 'delay'
    colors      = [
        'dimgray', 
        'orange', 
        'limegreen', 
        'red', 
        'blue'
    ]
    tick_labels = np.arange(6)
    xticks      = tick_labels * 2400

    plt.figure(figsize=(8, 4))
    for j, algorithm in enumerate(algorithms):
        # set 
        args.algorithm = algorithm
        args.n_node    = n_node
        args.n_rb      = int(n_node / 6)

        # load
        df = util.load(args)
        if 'centralize' in algorithm:
            y = df[metric].to_numpy()[:240 * 44]
            y = y.reshape(240, 44)
            y_front = y[:, 0].reshape(-1, 1)
            y = np.concatenate([y_front, y_front, y_front, y_front, y_front, y_front, y], axis=1)
            y = y.reshape(-1)
        else:
            y = df[metric].to_numpy()[:240 * 50]
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
    plt.savefig('figure/mean_time_12_2_10.svg')
    plt.show()
