import pandas as pd

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
