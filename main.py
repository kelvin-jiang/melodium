import argparse

from sinusoid_extraction import spectral_transform
from salience import compute_saliences
from utils import load_wav_file, write_wav_file


def extract_melody(audio, sampling_rate, n_workers):
    # sinusoid extraction
    # TODO: equal loudness filter
    f, t, zxx = spectral_transform(audio, sampling_rate)
    # TODO: frequency correction

    # compute saliences
    compute_saliences(f, t, zxx, n_workers)

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
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    sampling_rate, audio = load_wav_file(args.input)
    melody = extract_melody(audio, sampling_rate, args.workers)
    # write_wav_file(melody, args.output, sampling_rate)


if __name__ == '__main__':
    main()
