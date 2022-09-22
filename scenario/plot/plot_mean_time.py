import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from . import util

def plot_mean_time(args):
    algorithms = [
        # 'dql', 
        # 'random', 
        'static',
        'qmix', 
        'centralize', 
        'centralize_fast', 
        # 'optimal',
    ]

    labels = [
        # 'Distributed Q-learning',
        # 'Random',
        'RR',
        'QMIX',
        'WF (1s)',
        'WF (25ms)',
        # 'WF (known traffic distribution)'
    ]

    n_nodes     = [12, 24, 48, 96]
    metrics     = ['delay', 'fairness', 'delay', 'queue_length', 'traffic']
    fmts        = ['k.-', 'r.-', 'b.-', 'b-.']
    xticks      = [1, 3, 5, 7, 9]
    tick_labels = [1, 2, 4, 4, 5]

    n_step = int(30 / (0.5e-3 * 50))

    for n_node in n_nodes:
        for j, algorithm in enumerate(algorithms):
            metric    = metrics[0]

            # set 
            args.algorithm = algorithm
            args.n_node    = n_node
            args.n_rb      = int(n_node / 6)

            # load
            df = util.load(args)
            y = df[metric].to_numpy()
            mean_y = []
            for i in range(int(args.n_test_episode * args.n_step // n_step) + 1):
                mean_y.append(np.mean(y[i * n_step: (i + 1) * n_step]))
            mean_y = np.array(mean_y)
            # plot
            plt.plot(mean_y, fmts[j], 
                     label=labels[j], 
                     markersize=10
            )
        plt.ylabel('Average delay (ms)')
        plt.xlabel('Time (minutes)')
        plt.xticks(ticks=xticks, labels=tick_labels)
        # plt.legend()
        plt.savefig('figure/mean_time_12_2_10.svg')
        # plt.show()
        break
