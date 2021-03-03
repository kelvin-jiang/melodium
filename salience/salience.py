import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time


# number of quantization bins for F0 candidates
n_bins = 600
# number of harmonics considered (N_h in paper)
n_harmonics = 20
# harmonic weighting parameter (alpha in paper)
harmonic_weight_param = 0.8
# magnitude compression parameter (beta in paper)
magnitude_compression = 1
# maximum allowed difference (in dB) between a magnitude
# and the and the magnitude of the highest peak (gamma in paper)
max_magnitude_diff = 40
# quantization range in hz
min_freq = 55
max_freq = 1760


def get_bin_index(freq):
    """
    :param freq: frequency from 55Hz to 1759Hz
    :return: the quantized bin number [0,599]
    """
    if freq < min_freq or freq >= max_freq:
        print('error: frequency out of quantization range')
        sys.exit(1)

    return np.floor((1200 * np.log2(freq/55)) / 10)


def magnitude_threshold(magnitude, max_magnitude):
    db_diff = 20 * np.log10(max_magnitude / magnitude)
    return 1 if db_diff < max_magnitude_diff else 0


def weighting_function(b, h, f0):
    # distance in semitones between harmonic frequency and center frequency of bin b
    if f0 < min_freq or f0 >= max_freq:
        return 0
    d_semitones = abs(get_bin_index(f0) - b) / 10
    weighting = (np.cos(d_semitones * np.pi / 2) ** 2) * (harmonic_weight_param ** (h - 1))
    return weighting if d_semitones <= 1 else 0


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
            mag_threshold = magnitude_threshold(magnitude, max_magnitude)
            weighting = weighting_function(b, h, f0)
            if get_bin_index(f0) > b and weighting == 0:
                # stop early, because later frequencies will be too large
                break
            if weighting > 0 and not has_weight:
                # save the smallest activated frequency in this iteration
                # for optimization purposes
                has_weight = True
                first_freq_ind = i
            compressed_magnitude = magnitude ** magnitude_compression
            salience += mag_threshold * weighting * compressed_magnitude

    return salience


def compute_salience(f, peaks):
    salience = np.zeros(n_bins)
    max_magnitude = np.max(peaks)
    for b in range(n_bins):
        salience[b] = salience_function(b, f, peaks, max_magnitude)

    return salience


def plot_saliences(t, saliences):
    bins = np.arange(n_bins)
    # take the middle frequency in each bin
    frequencies = 55 * (2 ** ((10 * bins + 5)/1200))
    plt.pcolormesh(t, frequencies, saliences, shading='gouraud')
    plt.title('Salience')
    plt.ylabel('frequency (hz)')
    plt.ylim([0, 600])
    plt.xlabel('time (s)')
    plt.savefig('./output/salience', dpi=1024)


def compute_saliences(f, t, zxx):
    # take only magnitudes
    zxx = np.abs(zxx)
    t_size = 200
    saliences = np.zeros((n_bins, t_size))
    for i in range(t_size):
        start_time = time.time()
        magnitudes = zxx[:, i]
        # find peaks
        peak_indices, _ = find_peaks(magnitudes)

        # salience function
        salience = compute_salience(
            f[peak_indices], magnitudes[peak_indices])
        saliences[:, i] = salience
        print(f'{i+1}/{t_size}: {time.time() - start_time : .2f}s')

    plot_saliences(t[:t_size], saliences)
