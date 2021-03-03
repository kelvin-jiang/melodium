import argparse

from sinusoid_extraction import spectral_transform
from salience import compute_salience
from utils import load_wav_file, write_wav_file
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np


# number of quantization bins for F0 candidates
n_bins = 600


def plot_saliences(t, saliences):
    bins = np.arange(n_bins)
    plt.pcolormesh(t, bins, saliences, shading='gouraud')
    plt.savefig('./output/salience', dpi=1024)


def extract_melody(audio, sampling_rate):
    # sinusoid extraction
    # TODO: equal loudness filter
    f, t, zxx = spectral_transform(audio, sampling_rate)
    # TODO: frequency correction

    # take only magnitudes
    zxx = np.abs(zxx)
    t_size = 10
    saliences = np.zeros((n_bins, t_size))
    for i in range(t_size):
        print(f'{i+1}/{t_size}')
        magnitudes = zxx[:, i]
        # find peaks
        peak_indices, _ = find_peaks(magnitudes)

        # salience function
        salience = compute_salience(
            f[peak_indices], magnitudes[peak_indices])
        saliences[:, i] = salience

    plot_saliences(t[:t_size], saliences)

    # pitch contour creation
    # TODO: peak filtering
    # TODO: peak streaming
    # TODO: pitch contour characterization

    # melody selection
    # TODO: voicing detection
    # TODO: pitch outlier removal
    # TODO: melody peak selection
    return 'mELodY'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_audio_file', required=True)
    parser.add_argument('--output_melody_file', required=True)
    args = parser.parse_args()

    sampling_rate, audio = load_wav_file(args.input_audio_file)
    melody = extract_melody(audio, sampling_rate)
    # write_wav_file(melody, args.output_melody_file, sampling_rate)


if __name__ == '__main__':
    main()
