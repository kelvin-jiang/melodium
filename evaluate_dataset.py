import argparse
import glob
import numpy as np

from contour import create_contours, select_melody
from evaluation import evaluate_melody
from spectral import get_hz_from_bin
from spectral import hop_size
from utils import load_melody

def main():
    metrics = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--saliences_dir', required=True)
    parser.add_argument('--reference_dir', required=True)
    parser.add_argument('--reference_hop_size', type=int, required=True)
    args = parser.parse_args()

    fs = 44100

    salience_files = [f for f in sorted(glob.glob(f'{args.saliences_dir}/*')) if f.endswith('.npy')]
    for salience_file in salience_files:
        name = salience_file[salience_file.rfind('/')+1:-4]  # lol

        saliences = np.load(salience_file)
        contours, space = create_contours(saliences, fs)
        melody_bins = select_melody(contours, space.shape[1], fs)
        melody_freqs = get_hz_from_bin(melody_bins)

        reference_file = f'{args.reference_dir}/{name}REF.txt'
        melody_metrics = evaluate_melody(melody_freqs, hop_size, reference_file, args.reference_hop_size, 44100)
        metrics.append(melody_metrics)

        print(f'done evaluating {salience_file}!')

    metrics_avg = np.mean(np.array(metrics), axis=0)
    print(f'total voiced recall rate: {metrics_avg[0]: .3f}')
    print(f'total voiced false alarm rate: {metrics_avg[1]: .3f}')
    print(f'total raw pitch accuracy: {metrics_avg[2]: .3f}')
    print(f'total overall accuracy: {metrics_avg[3]: .3f}')

if __name__ == '__main__':
    main()
