from heapq import heappop, heappush
import matplotlib.pyplot as plt
import numpy as np

from contour import create_contours
from spectral import hop_size, fft_length, inverse_spectral_transform
from salience import get_hz_from_bin
from utils import write_wav_file

voicing_lenience = 0.2
octave_error_iters = 3
sliding_mean_filter_size = 5  # in seconds

def detect_voicing(contours):
    filtered_contours = []

    # calculate voicing threshold
    avg_contour_mean_salience = np.mean([contour.salience_mean for contour in contours])

    for contour in contours:
        if contour.has_vibrato() or contour.pitch_deviation > 40:
            continue

        voicing_threshold = avg_contour_mean_salience - voicing_lenience * contour.salience_deviation
        if contour.salience_mean >= voicing_threshold:
            filtered_contours.append(contour)

    return filtered_contours

def compute_melody_pitch_mean(contours, t_size, fs):
    melody_pitch_mean = np.zeros(t_size)
    melody_pitch_contours = np.zeros(t_size)
    for contour in contours:
        contour_t_end = contour.t_start + len(contour.bins)
        weights = contour.saliences / np.sum(contour.saliences)
        melody_pitch_mean[contour.t_start:contour_t_end] += weights @ contour.bins
        melody_pitch_contours[contour.t_start:contour_t_end] += 1

    # set indices where contours are not present to 1 to avoid dividing by zero
    melody_pitch_contours[melody_pitch_contours == 0] = 1

    melody_pitch_mean /= melody_pitch_contours

    # apply smoothing to melody pitch mean with sliding mean filter
    smoothed_melody_pitch_mean = np.zeros(t_size)
    filter_size = int(sliding_mean_filter_size // (hop_size / fs))
    for i in range(t_size):
        # note that end - start != filter_size since integer division, but will not be off by much
        start = max(i - filter_size // 2, 0)
        end = i + filter_size // 2
        window = melody_pitch_mean[start:end]
        smoothed_melody_pitch_mean[i] = np.mean(window[window != 0])  # only use non-zero values in window

    return smoothed_melody_pitch_mean

def compute_mean_distance(bins1, bins1_start, bins2, bins2_start):
    # note that t_start of bins1 <= t_start of bins2
    distances = []
    for i in range(bins2_start, bins2_start + len(bins2)):
        contour1_index = i - bins1_start
        if contour1_index >= len(bins1):
            # break if they don't overlap anymore
            break
        contour2_index = i - bins2_start
        distances.append(abs(bins1[contour1_index] - bins2[contour2_index]))

    return np.mean(distances)

def remove_octave_errors(contours, t_size, fs):
    filtered_contours = [contour for contour in contours]

    # sort contours by start frame to detect octave duplicates later
    contours.sort(key=lambda c: c.t_start)

    for _ in range(octave_error_iters):
        # compute melody pitch mean
        melody_pitch_mean = compute_melody_pitch_mean(filtered_contours, t_size, fs)

        # detect octave errors (consider all input contours)
        hq = []
        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                i_end = contours[i].t_start + len(contours[i].bins)
                overlap = min(i_end, contours[j].t_start + len(contours[j].bins)) - contours[j].t_start
                if overlap > 0:
                    mean_distance = compute_mean_distance(contours[i].bins, contours[i].t_start, contours[j].bins,
                                                          contours[j].t_start)
                    if 115 <= mean_distance <= 125:  # 1200 +/- 50 cents
                        heappush(hq, (-overlap, i, j))

        filtered = set()
        while len(hq) > 0:
            _, i, j = heappop(hq)
            if i in filtered or j in filtered:
                continue
            i_distance = compute_mean_distance(melody_pitch_mean, 0, contours[i].bins, contours[i].t_start)
            j_distance = compute_mean_distance(melody_pitch_mean, 0, contours[j].bins, contours[j].t_start)
            if i_distance > j_distance:
                filtered.add(i)
            else:
                filtered.add(j)
        filtered_contours = [contour for i, contour in enumerate(contours) if i not in filtered]

        # re-compute melody pitch mean
        melody_pitch_mean = compute_melody_pitch_mean(filtered_contours, t_size, fs)

        # remove pitch outliers (consider only non-octave error contours)
        non_outlier_contours = []
        for contour in filtered_contours:
            mean_distance = compute_mean_distance(melody_pitch_mean, 0, contour.bins, contour.t_start)
            if mean_distance < 120:  # 1200 cents
                non_outlier_contours.append(contour)
        filtered_contours = non_outlier_contours

    return filtered_contours

def select_melody(contours, t_size, fs):
    melody = np.zeros(t_size)

    contours = detect_voicing(contours)
    contours = remove_octave_errors(contours, t_size, fs)
    print(f'selected {len(contours)} melody contours')

    # determine melody
    melody_saliences = np.zeros(t_size)
    for contour in contours:
        for i in range(contour.t_start, min(contour.t_start + len(contour.bins), t_size - 1)):
            if contour.salience_total > melody_saliences[i]:
                melody[i] = contour.bins[i - contour.t_start]
                melody_saliences[i] = contour.salience_total

    return melody

def plot_melody(melody, fs, filename):
    plt.clf()

    tt = np.arange(len(melody)) * (hop_size / fs)
    nonzero_melody = np.copy(melody)
    nonzero_melody[nonzero_melody == 0] = np.nan
    plt.plot(tt, nonzero_melody, 'k')
    plt.title('Melody')
    plt.xlabel('time (s)')
    plt.ylabel('frequency (bins)')
    plt.savefig(filename, dpi=128, bbox_inches='tight')

def main():
    fs = 44100
    saliences = np.load('./output/salience.npy')
    contours, space = create_contours(saliences, fs)
    melody = select_melody(contours, space.shape[1], fs)
    hz_res = fs / fft_length
    melody_hz = (get_hz_from_bin(melody) // hz_res).astype(np.int_)
    f_size = fft_length // 2 + 1
    t_size = melody_hz.shape[0]
    melody_2d = np.zeros((t_size, f_size))
    for t in range(t_size):
        melody_2d[t, melody_hz[t]] = 1
    melody_2d = np.transpose(melody_2d)
    print(melody_2d)
    _, audio = inverse_spectral_transform(melody_2d, fs)
    write_wav_file(audio, './output/melody.wav', fs)
    plot_melody(melody, fs, './output/melody')

if __name__ == '__main__':
    main()
