import numpy as np
from scipy.signal import find_peaks
from heapq import *


peak_threshold = 0.9    # tau+
dev_degree = 0.9    # tau_sigma
pitch_cont_threshold = 80
max_gap_length = 0.1


def filter_saliences(saliences):
    t_size = saliences.shape[1]
    high, low = [set([]) for _ in range(t_size)], [set([]) for _ in range(t_size)]
    # contains all peak salience
    accum = []
    # stage 1
    for i in range(t_size):
        salience = saliences[:, 0]
        salience_peaks, _ = find_peaks(salience)
        max_salience = np.max(salience[salience_peaks])
        for b in salience_peaks:
            accum.append(salience[b])
            if salience[b] >= peak_threshold * max_salience:
                # only keep peaks that pass the threshold
                high[i].add(b)
            else:
                # filter out peaks that don't pass threshold
                low[i].add(b)

    accum = np.asarray(accum)
    mean, std = np.mean(accum), np.std(accum)
    hq = []

    # stage 2
    for i in range(t_size):
        for b in list(high[i]):
            sal = saliences[b, i]
            if sal < mean - dev_degree * std:
                # filter out salience that's lower than global threshold
                high[i].remove(b)
                low[i].add(b)
            else:
                # passed both filters, include it in priority queue
                hq.append((-sal, i, b))

    heapify(hq)
    return high, low, hq


def contour_creation():
    saliences = np.load('./data/salamon-salience-10s.npy')
    high, low, hq = filter_saliences(saliences)


if __name__ == '__main__':
    contour_creation()
