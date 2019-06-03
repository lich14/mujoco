import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--file', type=str, default="./csvfiles/loss.csv")
parser.add_argument('--num', type=int, default=1)
args = parser.parse_args()

print(args.file)
path = args.file
a = 2 / 13


def ema(N, Price):
    if N <= 1:
        return Price[1]
    return (1 - a) * ema(N - 1, Price) + a * Price[N]


with open(path, "rt") as f:
    csv_read = csv.reader(f)
    print(next(csv_read))
    print(csv_read)
    listnum = []
    plt.figure()
    for line in csv_read:
        listnum.append(float(line[0]))
    '''x = np.linspace(1, len(listnum), num=len(listnum), endpoint=True)
    f2 = interp1d(x, listnum, kind='cubic')
    x_n = np.linspace(1, len(listnum), num=4 * len(listnum), endpoint=True)'''
    show = np.array(listnum)
    print(np.shape(show))
    print(np.mean(show))
    N = 20
    n = np.ones(N)
    weights = n / N
    sma = np.convolve(weights, show)[N - 1:-N + 1]
    print(np.mean(show))
    t = np.arange(N - 1, len(show))
    plt.plot(t, sma, lw=1)
    #plt.plot(show, lw=0.01)

    plt.show()
