import matplotlib.pyplot as plt
import numpy as np

def legend_time(args):
    labels = [
        'RR',
        'QMIX',
        'WF (1s)',
        'WF (25ms)',
    ]
    fmts = ['k.-', 'r.-', 'b.-', 'b-.']
    x    = np.random.rand(10)
    f1 = plt.figure('plot')
    f2 = plt.figure('legend', figsize=(10, 6))
    ax = f1.add_subplot(111)
    lines = []
    for fmt, label in zip(fmts, labels):
        line, = ax.plot(x, fmt)
        lines.append(line)
    f2.legend(lines, labels, loc='center', ncol=4)
    f2.savefig('figure/legend_time.svg')
