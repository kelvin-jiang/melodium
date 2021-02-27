import argparse

from sinusoid_extraction import spectral_transform
from utils import load_wav_file, write_wav_file

def extract_melody(audio):
    # sinusoid extraction
    # TODO: equal loudness filter
    audio = spectral_transform(audio)
    # TODO: frequency correction

    # salience function
    # TODO: salience function computation

    # pitch contour creation
    # TODO: peak filtering
    # TODO: peak streaming
    # TODO: pitch contour characterization

    # melody selection
    # TODO: voicing detection
    # TODO: pitch outlier removal
    # TODO: melody peak selection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_audio_file', required=True)
    parser.add_argument('--output_melody_file', required=True)
    args = parser.parse_args()

    sampling_rate, audio = load_wav_file(args.input_audio_file)
    melody = extract_melody(audio)
    write_wav_file(melody, args.output_melody_file, sampling_rate)
