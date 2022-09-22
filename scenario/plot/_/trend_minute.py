import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# font = {'weight' : 'bold',
#         'size'   : 16}
# mpl.rc('font', **font)

def load(args):
    # field names
    names = ['t',
             'transmit',
             'success',
             'collision',
             'traffic',
             'drop',
             'queue_length',
             'delay',
             'good_put',
             'fairness',
    ]

    # construct folder name
    label  = f'{args.algorithm},{args.state},{args.reward}_{args.spatial}'
    label += f'_{args.n_rb},{args.n_node}_{int(args.n_step/40)}'
    path = f'{args.csv_dir}/{args.mode}/{label}.csv'
    df = pd.read_csv(path, names=names, sep=',')
    return df

def trend_minute(args):
    algorithms  = ['centralize', 'qmix', 'random', 'static']
    labels      = ['Centralized Load-balancing', 'QMIX', 'Random', 'RoundRobin']
    n_nodes     = [12, 24, 48, 96]
    metrics     = ['delay', 'fairness', 'delay', 'queue_length', 'traffic']
    fmts        = ['k-', 'ro-', 'k+-', 'bs-']
    xticks      = [0, 1, 2, 3, 4]
    tick_labels = [1, 2, 3, 4, 5]
    for n_node in n_nodes:
        for j, algorithm in enumerate(algorithms):
            metric    = metrics[0]

            # set 
            args.algorithm = algorithm
            args.n_node    = n_node
            args.n_rb      = int(n_node / 6)

            # load
            df = load(args)
            y = df[metric].to_numpy()
            mean_y = []
            for i in range(int(args.n_test_episode // 6)):    
                mean_y.append(np.mean(y[i * args.n_step * 6: (i + 1) * args.n_step * 6]))
            mean_y = np.array(mean_y)
            # show
            plt.plot(mean_y, fmts[j], label=labels[j], markersize=10)
        plt.ylabel('Average delay (ms)')
        plt.xlabel('Minutes')
        plt.xticks(ticks=xticks, labels=tick_labels)
        # plt.legend()
        plt.show()
