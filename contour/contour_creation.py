from heapq import heapify, heappop
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from spectral import hop_size
from salience import n_bins

peak_threshold = 0.9  # tau+ in paper
dev_degree = 0.9  # tau_sigma in paper
pitch_cont_threshold = 80
max_gap = 0.1

class Contour:
    def __init__(self, t_start, peaks, saliences, sampling_frequency):
        self.t_start = t_start
        self.peaks = np.array(peaks)
        self.saliences = np.array(saliences)
        self.sampling_frequency = sampling_frequency

        # compute characteristics
        self.pitch_mean = np.mean(self.peaks)
        self.pitch_deviation = np.std(self.peaks)
        self.salience_mean = np.mean(self.saliences)
        self.salience_total = np.sum(self.saliences)
        self.salience_deviation = np.std(self.saliences)
        self.length = len(self.peaks)
        self.vibrato = self.has_vibrato()

    def has_vibrato(self):
        return False

def filter_saliences(saliences):
    t_size = saliences.shape[1]
    high, low = [set([]) for _ in range(t_size)], [set([]) for _ in range(t_size)]
    # contains all peak salience
    accum = []

    # stage 1
    for i in range(t_size):
        salience = saliences[:, i]
        salience_peaks, _ = find_peaks(salience)
        max_salience = np.max(salience[salience_peaks])
        for b in salience_peaks:
            if salience[b] >= peak_threshold * max_salience:
                # only keep peaks that pass the threshold
                high[i].add(b)
                accum.append(salience[b])
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
    # find peak closest to b_target
    if b_target in peaks[i]:
        return b_target
    # max dist in number of bins
    max_dist = pitch_cont_threshold // 10
    for dist in range(1, max_dist + 1):
        for sign in [1, -1]:
            b = b_target + sign * dist
            if b in peaks[i]:
                return b
    return None

def track_salience(space, high, low, b_start, t_start, step, fs):
    contour = []
    t_size = space.shape[1]
    t = t_start
    t_last_high = t_start
    b_prev = b_start
    while True:
        t += step
        # check boundaries
        if t < 0 or t >= t_size:
            break

        # check gap
        t_unit = hop_size / fs
        gap = abs(t - t_last_high) * t_unit
        if gap > max_gap:
            # stop tracking
            break

        # try to find connecting peak from high
        b_high = find_connecting_peak(t, high, b_prev)
        if b_high is not None:
            # if peak found in high, remove from high and add to contours
            b_prev = b_high
            space[b_prev, t] = 1
            contour.append(b_prev)
            high[t].remove(b_prev)
            t_last_high = t
            continue

        # try to find ocnnecting peak from low
        b_prev = find_connecting_peak(t, low, b_prev)
        if b_prev is not None:
            # if peak found in low, remove from high and add to contours
            space[b_prev, t] = 1
            contour.append(b_prev)
            low[t].remove(b_prev)
            continue

        # end if cannot find connecting peak
        break

    return contour

def create_contours(saliences, fs):
    t_size = saliences.shape[1]
    # get S+, S-, and a max-heap of saliences in S+
    high, low, hq = filter_saliences(saliences)

    space = np.zeros((n_bins, t_size))
    contours = []
    while len(hq) > 0:
        start = heappop(hq)
        i, b = start[1], start[2]
        if b not in high[i]:
            # skip if this salience is removed from high
            continue
        # remove from high
        high[i].remove(b)
        space[b, i] = 1

        # track forward in time
        right_contour = track_salience(space=space, high=high, low=low, b_start=b, t_start=i, step=1, fs=fs)

        # track backwards in time
        left_contour = track_salience(space=space, high=high, low=low, b_start=b, t_start=i, step=-1, fs=fs)

        contour_peaks = left_contour[::-1] + [b] + right_contour
        t_start = i - len(left_contour)
        contour_saliences = [saliences[contour_peaks[i], t_start + i] for i in range(len(contour_peaks))]
        contour = Contour(t_start, contour_peaks, contour_saliences, fs / hop_size)
        contours.append(contour)

    print(f'created {len(contours)} contours')
    return contours, space

def plot_contours(space, fs, n_contours, filename):
    tt = np.arange(space.shape[1]).astype(np.float64) * (hop_size / fs)
    bins = np.arange(n_bins)
    plt.pcolormesh(tt, bins, space, shading='nearest', cmap='binary')
    plt.title(f'Contours with ELF ({n_contours} total)')
    plt.ylabel('frequency (bins)')
    plt.xlabel('time (s)')
    plt.savefig(filename, dpi=128)

def main():
    fs = 44100
    saliences = np.load('./output/salience.npy')
    contours, space = create_contours(saliences, fs)
    plot_contours(space, fs, len(contours), './output/contours')

if __name__ == '__main__':
    main()
