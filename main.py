import argparse

from contour import create_contours, plot_contours
from salience import compute_saliences
from spectral import equal_loudness_filter, spectral_transform, H
from utils import load_wav_file, write_wav_file

def extract_melody(audio, fs, n_workers):
    # sinusoid extraction
    audio = equal_loudness_filter(audio)
    f, t, Zxx = spectral_transform(audio, fs)
    # TODO: frequency correction

    # compute saliences
    saliences = compute_saliences(f, t, Zxx, n_workers, fs)

    # pitch contour creation
    contours, space, count = create_contours(saliences, fs)
    plot_contours(space, H / fs, './output/contours', f'Contours with ELF ({count} total)')

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

    fs, audio = load_wav_file(args.input)
    melody = extract_melody(audio, fs, args.workers)
    # write_wav_file(melody, args.output, fs)

if __name__ == '__main__':
    main()
