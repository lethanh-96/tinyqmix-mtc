import matplotlib.pyplot as plt
import numpy as np

def plot_flops(args):
    flops = {
        'qmix_lstm': [1298 + 3, 1476 + 3, 4936 + 3, 17808 + 3],
        'qmix': [58 + 8 + 9, 92 + 12 + 9, 312 + 20 + 9, 1136 + 36 + 9],
        'random': [2, 4, 8, 16],
    }

    for key in flops.keys():
        plt.plot(flops[key], label=key)
    plt.ylabel('FLOPs')
    plt.xlabel('number of devices')
    # plt.yscale('log')
    plt.xticks(ticks=np.arange(4), labels=[12, 24, 48, 96])
    plt.legend()
    plt.show()

    print(flops)