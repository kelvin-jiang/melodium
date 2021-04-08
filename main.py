import argparse
import numpy as np

from contour import create_contours, plot_contours, plot_melody, select_melody
from salience import compute_saliences, plot_saliences
from spectral import equal_loudness_filter, plot_spectral_transform, spectral_transform, hop_size, inverse_spectral_transform
from utils import load_wav_file, write_wav_file

def extract_melody(audio, fs, args):
    # sinusoid extraction
    if not args.omit_elf:
        print("Applying equal loudness filter...")
        audio = equal_loudness_filter(audio)
    print("Spectral transform...")
    f, t, Zxx = spectral_transform(audio, fs)
    print("Plotting spectral transform...")
    plot_spectral_transform(f, t, Zxx, fs, args.spectral_plot)
    # TODO: frequency correction

    # compute saliences
    t_size = t.shape[0] if args.duration < 0 else -(-args.duration * fs) // hop_size
    if args.use_cached_saliences:
        print("Loading salience...")
        saliences = np.load(args.cached_saliences)
    else:
        print("Computing salience...")
        saliences = compute_saliences(f, t, Zxx, args.workers, fs, args.cached_saliences, t_size)
    print("Plotting salience...")
    plot_saliences(t, saliences, fs, args.salience_plot, t_size)

    # pitch contour creation
    print("Creating pitch contours...")
    contours, space = create_contours(saliences, fs)
    print("Plotting pitch contours...")
    plot_contours(space, fs, len(contours), args.contour_plot)

    # melody selection
    print("Selecting melody...")
    melody = select_melody(contours, space.shape[1], fs)
    print("Plotting melody...")
    plot_melody(melody, fs, args.melody_plot)

    # reconstruct melody as audio signal
    _, melody_audio = inverse_spectral_transform(melody, fs)

    return melody, melody_audio

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
    parser.add_argument('--melody_plot', default='./output/melody')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--duration', type=int, default=-1)
    args = parser.parse_args()

    fs, audio = load_wav_file(args.input)
    assert fs == 44100
    melody, melody_audio = extract_melody(audio, fs, args)
    write_wav_file(melody_audio, args.output, fs)

if __name__ == '__main__':
    main()
