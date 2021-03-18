import numpy as np

from spectral import window_length

voicing_lenience = 0.2
octave_error_iters = 3
sliding_mean_filter = 5  # in seconds

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

def compute_melody_pitch_mean(contours, t_size):
    melody_pitch_mean = np.zeros(t_size)

    melody_pitch_contours = np.zeros(t_size)
    for contour in contours:
        contour_t_end = contour.t_start + len(contour.peaks)
        melody_pitch_mean[contour.t_start:contour_t_end] += contour.saliences @ contour.peaks
        melody_pitch_contours[contour.t_start:contour_t_end] += 1

    return melody_pitch_mean / melody_pitch_contours

def compute_mean_distance(contour1, contour2):
    # note that t_start of contour1 <= t_start of contour2
    distances = []
    for i in range(contour2.t_start, contour2.t_start + len(contour2.peaks)):
        contour1_index = i - contour1.t_start
        if contour1_index >= len(contour1.peaks):
            # break if they don't overlap anymore
            break
        contour2_index = i - contour2.t_start
        distances.append(abs(contour1.peaks[contour1_index] - contour2.peaks[contour2_index]))

    return np.mean(distances)

def remove_octave_errors(contours, t_size, fs):
    filtered_contours = [contour for contour in contours]

    # sort contours by start frame to detect octave duplicates later
    contours.sort(key=lambda c: c.t_start)

    for _ in range(octave_error_iters):
        # compute melody pitch mean
        melody_pitch_mean = compute_melody_pitch_mean(filtered_contours, t_size)

        # smooth melody pitch mean with sliding mean filter
        filtered_melody_pitch_mean = np.zeros(t_size)
        filter_size = sliding_mean_filter * fs // window_length
        for i in range(t_size):
            # end - start != filter_size but will not be off by much
            start = i - filter_size / 2
            end = i + filter_size / 2
            filtered_melody_pitch_mean[i] = np.mean(melody_pitch_mean[start:end])

        # detect octave duplicates
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
                mean_distance = compute_mean_distance(contours[i], contours[j])
                if 1150 <= mean_distance <= 1250:
                    i_distance = compute_mean_distance(filtered_melody_pitch_mean, contours[i])
                    j_distance = compute_mean_distance(filtered_melody_pitch_mean, contours[j])
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

    return contours

def select_melody(contours, t_size, fs):
    melody = np.zeros(t_size)

    contours = detect_voicing(contours)
    contours = remove_octave_errors(contours, t_size, fs)

    return melody
