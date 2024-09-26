import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def real2pred_length_plot():
    df = pd.read_csv("result.csv", sep=',')
    real = df['real']
    pred = df['prediction']

    x_axis = np.linspace(0, max(real), 100) # start, stop, num
    y_axis = np.linspace(0, max(real), 100)
    plt.plot(real, pred, 'o')
    plt.plot(x_axis, y_axis, linewidth=1)
    plt.show()

def entropy2syndrome():
    df = pd.read_csv("result.csv", sep=',')
    real = df['real']
    pred = df['prediction']
    entropy = df['entropy']
    N = max(real)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    plt.gca().set_ylim(bottom=0, top=1.2 * N)
    x_axis = np.linspace(0.01, 1, 100)
    ax.scatter(entropy, real, color='black', s=2)
    plt.show()
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rp', action='store_true')
    parser.add_argument('--es', action='store_true')
    args = parser.parse_args()

    if args.rp:
        real2pred_length_plot()
    elif args.es:
        entropy2syndrome()
