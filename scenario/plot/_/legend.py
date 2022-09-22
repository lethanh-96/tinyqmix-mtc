import matplotlib.pyplot as plt
import numpy as np

def legend(args):
    labels = ['Centralized Load-balancing', 'QMIX', 'Random', 'RoundRobin']
    fmts   = ['k-', 'ro-', 'k+-', 'bs-']
    x      = np.random.rand(10)
    f1 = plt.figure('plot')
    f2 = plt.figure('legend', figsize=(10, 6))
    ax = f1.add_subplot(111)
    lines = []
    for fmt, label in zip(fmts, labels):
        line, = ax.plot(x, fmt)
        lines.append(line)
    f2.legend(lines, labels, loc='center', ncol=2)
    f2.savefig('legend.png')
    # f2.show()