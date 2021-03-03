import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time
import multiprocessing as mp
from sinusoid_extraction import H   # hop size
import math


# number of quantization bins for F0 candidates
n_bins = 600
# number of harmonics considered (N_h in paper)
n_harmonics = 20
# harmonic weighting parameter (alpha in paper)
harmonic_weight = 0.8
# magnitude compression parameter (beta in paper)
# magnitude_compression = 1
# maximum allowed difference (in dB) between a magnitude
# and the and the magnitude of the highest peak (gamma in paper)
max_magnitude_diff = 40
# quantization range in hz
min_freq = 55
max_freq = 1760


harmonic_weights = []


def compute_harmonic_weight():
    harmonic_weights.append(1)
    for i in range(1, n_harmonics):
        harmonic_weights.append(harmonic_weights[i-1] * harmonic_weight)


def get_bin_index(freq):
    """
    :param freq: frequency from 55Hz to 1759Hz
    :return: the quantized bin number [0,599]
    """
    return (1200 * math.log2(freq/55)) // 10


def magnitude_threshold(magnitude, max_magnitude):
    db_diff = 20 * math.log10(max_magnitude / magnitude)
    return 1 if db_diff < max_magnitude_diff else 0


def weighting_function(b, h, b_f0):
    # distance in semitones between harmonic frequency and center frequency of bin b
    d_semitones = abs(b_f0 - b) / 10
    if d_semitones > 1:
        return 0
    weight = math.cos(d_semitones * math.pi / 2)
    return weight * weight * harmonic_weights[h-1]


def salience_function(b, f, peaks, max_magnitude):
    """
    :param b: F0 candidate
    :param f:
    :param peaks:
    :return: salience for freq
    """
    if f.shape[0] != peaks.shape[0]:
        print("error: f and peaks have mismatched length")
        sys.exit(1)

    salience = 0.0

    first_freq_ind = 0
    for h in range(1, n_harmonics + 1):
        has_weight = False
        # since our harmonics and frequencies are sorted, so only need to look for frequencies
        # that are greater than smallest activated frequency in the last harmonic h
        for i in range(first_freq_ind, peaks.shape[0]):
            magnitude = peaks[i]
            freq = f[i]
            f0 = freq / h
            if magnitude <= 0 or f0 < min_freq or f0 >= max_freq:
                # skip out of bounds frequencies
                continue
            b_f0 = get_bin_index(f0)
            weighting = weighting_function(b, h, b_f0)
            if weighting == 0 and b_f0 > b:
                # stop early, because later frequencies will be too large
                break
            if weighting > 0 and not has_weight:
                # save the smallest activated frequency in this iteration
                # for optimization purposes
                has_weight = True
                first_freq_ind = i
            mag_threshold = magnitude_threshold(magnitude, max_magnitude)
            salience += mag_threshold * weighting * magnitude

    return salience


def compute_salience(magnitudes, f, i):
    start_time = time.time()
    # find peaks
    peak_indices, _ = find_peaks(magnitudes)
    peaks, peak_f = magnitudes[peak_indices], f[peak_indices]

    salience = np.zeros(n_bins)
    max_magnitude = np.max(peaks)

    for b in range(n_bins):
        salience[b] = salience_function(b, peak_f, peaks, max_magnitude)

    print(f'frame {i}: {time.time() - start_time : .2f}s')
    return i, salience


def plot_saliences(t, saliences):
    bins = np.arange(n_bins)
    # take the middle frequency in each bin
    # frequencies = 55 * (2 ** ((10 * bins + 5)/1200))
    plt.pcolormesh(t, bins, saliences, shading='gouraud')
    plt.title('Salience')
    plt.ylabel('frequency (bins)')
    plt.xlabel('time (s)')
    plt.savefig('./output/salience', dpi=1024)


def compute_saliences(f, t, zxx, n_workers, sampling_rate, t_seconds=10):
    start_time = time.time()

    # take only magnitudes
    zxx = np.abs(zxx)
    # number of time samples (frames) in t_seconds (using ceil)
    t_size = -(-t_seconds * sampling_rate) // H
    saliences = np.zeros((n_bins, t_size))

    # pre-compute harmonic weight
    compute_harmonic_weight()

    # use multiprocessing to compute salience
    pool = mp.Pool(processes=n_workers)
    jobs = [(zxx[:, i], f, i) for i in range(t_size)]
    results = pool.starmap(compute_salience, jobs)
    pool.close()
    pool.join()

    for i, salience in results:
        saliences[:, i] = salience

    print(f'salience done: {time.time() - start_time : .2f}s')
    plot_saliences(t[:t_size], saliences)
    np.save('./output/salience', saliences)
