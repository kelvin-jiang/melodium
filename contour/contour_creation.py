import numpy as np
from scipy.signal import find_peaks
from heapq import *
# from sinusoid_extraction.spectral_transform import H    # hop size
# from salience import n_bins
import matplotlib.pyplot as plt


H = 128
n_bins = 600
peak_threshold = 0.9    # tau+
dev_degree = 0.9    # tau_sigma
pitch_cont_threshold = 80
max_gap = 0.1


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


def find_connecting_peak(i, peaks, b_target):
    for _, b in enumerate(peaks[i]):
        if abs(b - b_target) * 10 < 80:
            return b
    return None


def track_salience(contours, high, low, b_start, t_start, step, sampling_rate):
    t_size = contours.shape[1]
    t = t_start
    t_last_high = t_start
    b_prev = b_start
    while True:
        t += step
        # check boundaries
        if t < 0 or t >= t_size:
            break

        # check gap
        t_unit = H / sampling_rate
        gap = abs(t - t_last_high) * t_unit
        if gap > max_gap:
            # stop tracking
            break

        # try to find connecting peak from high
        b_high = find_connecting_peak(t, high, b_prev)
        if b_high is not None:
            # if peak found in high, remove from high and add to contours
            b_prev = b_high
            contours[b_prev, t] = 1
            high[t].remove(b_prev)
            t_last_high = t
            continue

        # try to find ocnnecting peak from low
        b_prev = find_connecting_peak(t, low, b_prev)
        if b_prev is not None:
            # if peak found in low, remove from high and add to contours
            contours[b_prev, t] = 1
            low[t].remove(b_prev)
            continue
        break


def plot_contours(contours, t_unit):
    t_size = contours.shape[1]
    tt = np.arange(t_size).astype(np.float)
    tt = tt * (H / t_unit)
    bins = np.arange(n_bins)
    plt.pcolormesh(tt, bins, contours, shading='gouraud', cmap='binary')
    plt.title('Contours')
    plt.ylabel('frequency (bins)')
    plt.xlabel('time (s)')
    plt.savefig('./output/contours', dpi=1024)


def contour_creation(sampling_rate):
    saliences = np.load('./data/salamon-salience-10s.npy')
    t_size = saliences.shape[1]
    high, low, hq = filter_saliences(saliences)

    contours = np.zeros((n_bins, t_size))
    while len(hq) > 0:
        start = heappop(hq)
        i, b = start[1], start[2]
        if b not in high[i]:
            # skip if this salience is removed from high
            continue
        # add to contours
        contours[b, i] = 1

        # track forward in time
        track_salience(
            contours=contours, high=high, low=low,
            b_start=b, t_start=i, step=1,
            sampling_rate=sampling_rate
        )

        # track backwards in time
        track_salience(
            contours=contours, high=high, low=low,
            b_start=b, t_start=i, step=1,
            sampling_rate=sampling_rate
        )
    plot_contours(contours, H / sampling_rate)


if __name__ == '__main__':
    contour_creation(44100)
