import matplotlib.pyplot as plt
import numpy as np

from contour import create_contours
from spectral import hop_size

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
        contour_t_end = contour.t_start + len(contour.peaks)
        melody_pitch_mean[contour.t_start:contour_t_end] += contour.saliences @ contour.peaks
        melody_pitch_contours[contour.t_start:contour_t_end] += 1

    # set indices where contours are not present to 1 to avoid dividing by zero
    melody_pitch_contours[melody_pitch_contours == 0] = 1
    melody_pitch_mean /= melody_pitch_contours

    # smooth melody pitch mean with sliding mean filter
    smoothed_melody_pitch_mean = np.zeros(t_size)
    filter_size = int(sliding_mean_filter_size // (hop_size / fs))
    for i in range(t_size):
        # end - start != filter_size but will not be off by much
        start = max(i - filter_size // 2, 0)
        end = i + filter_size // 2
        smoothed_melody_pitch_mean[i] = np.mean(melody_pitch_mean[start:end])

    return smoothed_melody_pitch_mean

def compute_mean_distance(peaks1, peaks1_start, peaks2, peaks2_start):
    # note that t_start of peaks1 <= t_start of peaks2
    distances = []
    for i in range(peaks2_start, peaks2_start + len(peaks2)):
        contour1_index = i - peaks1_start
        if contour1_index >= len(peaks1):
            # break if they don't overlap anymore
            break
        contour2_index = i - peaks2_start
        distances.append(abs(peaks1[contour1_index] - peaks2[contour2_index]))

    return np.mean(distances)

def remove_octave_errors(contours, t_size, fs):
    filtered_contours = [contour for contour in contours]

    # sort contours by start frame to detect octave duplicates later
    contours.sort(key=lambda c: c.t_start)

    for _ in range(octave_error_iters):
        # compute melody pitch mean
        melody_pitch_mean = compute_melody_pitch_mean(filtered_contours, t_size, fs)

        # detect octave errors (consider all input contours)
        filtered_contours.clear()
        i = 0
        while i < len(contours):
            remove_curr = False
            remove_next = False
            for j in range(i + 1, len(contours)):
                i_end = contours[i].t_start + len(contours[i].peaks)
                if i_end <= contours[j].t_start:
                    # break if i-th contour doesn't overlap with any other contour
                    break
                mean_distance = compute_mean_distance(contours[i].peaks, contours[i].t_start, contours[j].peaks,
                                                      contours[j].t_start)
                if 1150 <= mean_distance <= 1250:
                    i_distance = compute_mean_distance(melody_pitch_mean, 0, contours[i].peaks, contours[i].t_start)
                    j_distance = compute_mean_distance(melody_pitch_mean, 0, contours[j].peaks, contours[j].t_start)
                    if i_distance > j_distance:
                        remove_curr = True
                    else:
                        remove_next = True
                    break

            if not remove_curr:
                filtered_contours.append(contours[i])
            if remove_next:
                # remove next by skipping it
                i += 1
            i += 1

        # re-compute melody pitch mean
        melody_pitch_mean = compute_melody_pitch_mean(filtered_contours, t_size, fs)

        # remove pitch outliers (consider only non-octave error contours)
        non_outlier_contours = []
        for contour in filtered_contours:
            mean_distance = compute_mean_distance(melody_pitch_mean, 0, contour.peaks, contour.t_start)
            if mean_distance < 1200:
                non_outlier_contours.append(contour)
        filtered_contours = non_outlier_contours

    return contours

def select_melody(contours, t_size, fs):
    melody = np.zeros(t_size)

    contours = detect_voicing(contours)
    contours = remove_octave_errors(contours, t_size, fs)

    # determine melody
    melody_saliences = np.zeros(t_size)
    for contour in contours:
        for i in range(contour.t_start, min(contour.t_start + len(contour.peaks), t_size - 1)):
            if contour.salience_total > melody_saliences[i]:
                melody[i] = contour.peaks[i - contour.t_start]
                melody_saliences[i] = contour.salience_total

    return melody

def plot_melody(melody, fs, filename):
    tt = np.arange(len(melody)) * (hop_size / fs)
    nonzero_melody = np.copy(melody)
    nonzero_melody[nonzero_melody == 0] = np.nan
    plt.plot(tt, nonzero_melody, 'k')
    plt.title('Melody')
    plt.xlabel('time (s)')
    plt.ylabel('frequency (bins)')
    plt.savefig(filename, dpi=128)

def main():
    fs = 44100
    saliences = np.load('./output/salience.npy')
    contours, space = create_contours(saliences, fs)
    melody = select_melody(contours, space.shape[1], fs)

    plot_melody(melody, fs, './output/melody')

if __name__ == '__main__':
    main()
