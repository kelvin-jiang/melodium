import argparse
import matplotlib.pyplot as plt
import numpy as np

from utils import load_melody

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--reference_hop_size', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    fs = 44100
    reference = load_melody(args.reference)
    reference[reference == 0] = -np.inf

    plt.clf()
    tt = np.arange(len(reference)) * (args.reference_hop_size / fs)
    plt.plot(tt, reference, 'k')
    plt.title('Reference')
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    plt.savefig(args.output, dpi=128, bbox_inches='tight')

if __name__ == '__main__':
    main()
