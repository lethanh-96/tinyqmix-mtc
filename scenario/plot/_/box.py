import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def box(args):
    algorithms = ['optimal', 'centralize', 'centralize_fast', 'dql', 'qmix', 'random', 'static']
    n_nodes    = [12, 24, 48, 96]
    metrics    = ['delay', 'queue_length', 'traffic', 'fairness']

    for n_node in n_nodes:
        sequences = []
        for algorithm in algorithms:
            metric    = metrics[0]
            # set 
            args.algorithm = algorithm
            args.n_node    = n_node
            args.n_rb      = int(n_node / 6)
            # load
            df = load(args)
            y = df[metric].to_numpy()
            sequences.append(y)

        plt.boxplot(sequences, labels=algorithms, showfliers=False)
        plt.title(f'{n_node} nodes')
        plt.show()

    # FORMAT = f'& & & & & & & {np.mean(y)} & {np.std(y)} & {np.max(y)} \\'
