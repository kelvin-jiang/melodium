import argparse

from sinusoid_extraction import equal_loudness_filter, spectral_transform, H
from salience import compute_saliences
from utils import load_wav_file, write_wav_file
from contour import create_contours, plot_contours

def extract_melody(audio, sampling_rate, n_workers):
    # sinusoid extraction
    audio = equal_loudness_filter(audio)
    f, t, Zxx = spectral_transform(audio, sampling_rate)
    # TODO: frequency correction

    # compute saliences
    saliences = compute_saliences(f, t, Zxx, n_workers, sampling_rate)

    # pitch contour creation
    contours, space, count = create_contours(saliences, sampling_rate)
    plot_contours(space, H / sampling_rate, './output/contours', f'Contours with ELF ({count} total)')

    # melody selection
    # TODO: voicing detection
    # TODO: pitch outlier removal
    # TODO: melody peak selection
    return 'mELodY'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    sampling_rate, audio = load_wav_file(args.input)
    melody = extract_melody(audio, sampling_rate, args.workers)
    # write_wav_file(melody, args.output, sampling_rate)

if __name__ == '__main__':
    main()
