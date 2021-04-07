import argparse
cimport libc.math as math
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from scipy.signal import find_peaks
import sys
import time
cimport numpy as np
import cython

from spectral import equal_loudness_filter, hop_size, spectral_transform
from utils import load_wav_file

DTYPE = np.double
ctypedef np.double_t DTYPE_t

n_bins = 600  # number of quantization bins for F0 candidates
n_harmonics = 20  # number of harmonics considered (N_h in paper)
cdef int n_harmonics_c = n_harmonics
harmonic_weight = 0.8  # harmonic weighting parameter (alpha in paper)
# magnitude_compression = 1  # magnitude compression parameter (beta in paper)
max_magnitude_diff = 40  # maximum difference (in dB) between a magnitude and magnitude of highest peak (gamma in paper)
cdef double max_magnitude_diff_c = max_magnitude_diff
# quantization range in Hz
min_freq = 55
max_freq = 1760
cdef double min_freq_c = min_freq
cdef double max_freq_c = max_freq

harmonic_weights = [harmonic_weight**i for i in range(n_harmonics)]

def get_bin_index(double freq):
    """
    :param freq: frequency from 55Hz to 1759Hz
    :return: the quantized bin number [0,599]
    """
    cdef int result
    result = int(math.floor((1200 * math.log2(freq / 55)) // 10))
    return result

def get_hz_from_bin(b):
    return 55 * (2 ** ((10 * b + 5) / 1200))

@cython.cdivision(True)
def magnitude_threshold(double magnitude, double max_magnitude):
    cdef double db_diff
    cdef int result
    cdef double max_diff = max_magnitude_diff_c
    db_diff = 20 * math.log10(max_magnitude / magnitude)
    result = 1 if db_diff < max_diff else 0
    return result

def weighting_function(int b, int h, double b_f0):
    # distance in semitones between harmonic frequency and center frequency of bin b
    cdef double result
    d_semitones = abs(b_f0 - b) / 10
    if d_semitones > 1:
        return 0
    weight = math.cos(d_semitones * math.pi / 2)
    result = weight * weight * harmonic_weights[h - 1]
    return result

@cython.cdivision(True)
def salience_function(int b, np.ndarray[DTYPE_t] f, np.ndarray[DTYPE_t] peaks, double max_magnitude):
    """
    :param b: F0 candidate
    :param f:
    :param peaks:
    :return: salience for freq
    """
    if f.shape[0] != peaks.shape[0]:
        print('error: f and peaks have mismatched length')
        sys.exit(1)

    cdef DTYPE_t magnitude
    cdef DTYPE_t freq
    cdef double f0
    cdef int b_f0
    cdef double weighting
    cdef double mag_threshold
    cdef double salience = 0.0
    cdef int first_freq_ind = 0
    cdef double max_frequency = max_freq_c
    cdef double min_frequency = min_freq_c
    cdef int num_harmonics = n_harmonics_c
    cdef int h

    for h in range(1, num_harmonics + 1):
        has_weight = False
        # since our harmonics and frequencies are sorted, so only need to look for frequencies
        # that are greater than smallest activated frequency in the last harmonic h
        for i in range(first_freq_ind, peaks.shape[0]):
            magnitude = peaks[i]
            freq = f[i]
            f0 = freq / h
            if magnitude <= 0 or f0 < min_frequency or f0 >= max_frequency:
                # skip out of bounds frequencies
                continue
            b_f0 = get_bin_index(f0)
            weighting = weighting_function(b, h, b_f0)
            if weighting == 0 and b_f0 > b:
                # stop early, because later frequencies will be too large
                break
            if weighting > 0 and not has_weight:
                # save the smallest activated frequency in this iteration for optimization purposes
                has_weight = True
                first_freq_ind = i
            mag_threshold = magnitude_threshold(magnitude, max_magnitude)
            salience += mag_threshold * weighting * magnitude

    return salience

def compute_frame_salience(np.ndarray[DTYPE_t] magnitudes, np.ndarray[DTYPE_t] f, int i):
    cdef np.ndarray[DTYPE_t] peaks, peak_f, salience
    cdef double max_magnitude

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

def compute_saliences(f, t, Zxx, n_workers, fs, cached_saliences, t_size):
    start_time = time.time()

    # take only magnitudes
    Zxx = np.abs(Zxx)

    # number of time samples (frames) in t_seconds (using ceil)
    saliences = np.zeros((n_bins, t_size))

    # use multiprocessing to compute salience
    pool = mp.Pool(processes=n_workers)
    jobs = [(Zxx[:, i], f, i) for i in range(t_size)]
    results = pool.starmap(compute_frame_salience, jobs, chunksize=100)
    pool.close()
    pool.join()

    for i, salience in results:
        saliences[:, i] = salience

    print(f'salience done: {time.time() - start_time : .2f}s')

    np.save(cached_saliences, saliences)

    return saliences

def plot_saliences(t, saliences, fs, filename, t_size):
    plt.clf()

    bins = np.arange(n_bins)
    # take the middle frequency in each bin
    # frequencies = 55 * (2 ** ((10 * bins + 5)/1200))
    plt.pcolormesh(t[:t_size], bins, saliences, shading='gouraud', cmap='hot')
    plt.title('Salience')
    plt.xlabel('time (s)')
    plt.ylabel('frequency (bins)')
    plt.savefig(filename, dpi=128, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='./output/salience.npy')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    fs, audio = load_wav_file(args.input, merge_channels=True)

    audio = equal_loudness_filter(audio)
    f, t, Zxx = spectral_transform(audio, fs)

    saliences = compute_saliences(f, t, Zxx, args.workers, fs, args.output)
    plot_saliences(t, saliences, fs, './output/salience')

if __name__ == '__main__':
    main()
