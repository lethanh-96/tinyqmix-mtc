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

def table(args):
    algorithms = ['dql', 
                  'random', 
                  'static',
                  'centralize', 
                  'qmix', 
                  'centralize_fast', 
                  'optimal',
                  ]
    labels     = ['Distributed Q-Learning',
                  'Random',
                  'Round Robin',
                  'Water-filling (1s)',
                  'QMIX',
                  'Water-filling (25ms)',
                  'Water-filling + Known traffic distribution',
                  ]
    labels = dict(zip(algorithms, labels))
    n_nodes    = [12, 24, 48, 96]
    metrics    = ['delay', 'queue_length', 'traffic', 'fairness']

    for n_node in n_nodes:
        print(f'{n_node}/{int(n_node//6)}')
        for algorithm in algorithms:
            metric    = metrics[0]
            # set 
            args.algorithm = algorithm
            args.n_node    = n_node
            args.n_rb      = int(n_node / 6)
            # load
            df = load(args)
            y = df[metric].to_numpy()

            line = f'& {labels[algorithm]} & & & & & & & {np.mean(y).round(2)} & {np.std(y).round(2)} & {np.max(y).round(2)} \\\\'
            print(line)
        print('\\midrule')