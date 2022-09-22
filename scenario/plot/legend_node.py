import matplotlib.pyplot as plt
import numpy as np

def legend_node(args):
    labels = [
        'DQL',
        'Random',
        'RR',
        'WF',
        'LSTMQMIX',
        'TinyQMIX',
        'WFLB'
    ]
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
    r  = np.arange(10)
    x  = np.random.rand(10)
    f1 = plt.figure('plot')
    f2 = plt.figure('legend', figsize=(10, 6))
    ax = f1.add_subplot(111)
    bars = []
    for i, label in enumerate(labels):
        bar = ax.bar(r, x, fc=fcs[i], ec=ecs[i], hatch=hatches[i])
        bars.append(bar)
    f2.legend(bars, labels, loc='center', ncol=4)
    f2.savefig('figure/legend_node.svg')
