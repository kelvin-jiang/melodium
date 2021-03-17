import argparse
import numpy as np

from contour import create_contours, plot_contours
from salience import compute_saliences, plot_saliences
from spectral import equal_loudness_filter, plot_spectral_transform, spectral_transform
from utils import load_wav_file, write_wav_file

def extract_melody(audio, fs, args):
    # sinusoid extraction
    if not args.omit_elf:
        audio = equal_loudness_filter(audio)
    f, t, Zxx = spectral_transform(audio, fs)
    plot_spectral_transform(f, t, Zxx, fs, args.spectral_plot)
    # TODO: frequency correction

    # compute saliences
    if args.use_cached_saliences:
        saliences = np.load(args.cached_saliences)
    else:
        saliences = compute_saliences(f, t, Zxx, args.workers, fs, args.cached_saliences)
    plot_saliences(t, saliences, fs, args.salience_plot)

    # pitch contour creation
    contours, space, count = create_contours(saliences, fs)
    plot_contours(space, fs, len(contours), args.contour_plot)

    # melody selection
    # TODO: voicing detection
    # TODO: pitch outlier removal
    # TODO: melody peak selection
    return 'mELodY'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--omit_elf', action='store_true')
    parser.add_argument('--spectral_plot', default='./output/spectral')
    parser.add_argument('--use_cached_saliences', action='store_true')
    parser.add_argument('--cached_saliences', default='./output/salience.npy')
    parser.add_argument('--salience_plot', default='./output/salience')
    parser.add_argument('--contour_plot', default='./output/contours')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    fs, audio = load_wav_file(args.input)
    # assert fs == 44100
    melody = extract_melody(audio, fs, args)
    # write_wav_file(melody, args.output, fs)

if __name__ == '__main__':
    main()
