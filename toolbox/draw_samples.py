import matplotlib.pyplot as plt
import numpy as np


def plot_points(data):
    plt.figure(figsize=[16, 4])
    plt.gca().invert_yaxis()
    plt.axis('equal')
    pts = np.array(data).cumsum(axis=0)
    data[-1][-1] = 1
    idx = [i for i, v in enumerate(data) if data[i][-1] == 1]
    start = 0
    for end in idx:
        tmp = pts[start:end + 1]
        plt.plot(tmp[:, 0], tmp[:, 1], linewidth=2)
        start = end + 1
    plt.show()
