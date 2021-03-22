import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, stft

from utils import load_wav_file

window_length = 2048  # M in paper
hop_size = 128  # H in paper
fft_length = 8192  # N in paper

def spectral_transform(audio, fs):
    f, t, Zxx = stft(audio, fs, nperseg=window_length, noverlap=window_length - hop_size, nfft=fft_length)

    return f, t, Zxx

def plot_spectral_transform(f, t, Zxx, fs, filename, t_seconds=10, f_freqs=2000):
    plt.clf()

    # peak finding (for each time window)
    for i in range(Zxx.shape[1]):
        peaks, _ = find_peaks(np.abs(Zxx[:, i]))
        non_peaks = np.setdiff1d(np.arange(Zxx.shape[0]), peaks)
        np.put(Zxx[:, i], non_peaks, 0)

    # apply sign function on magnitudes to make peaks visible
    Zxx = np.sign(np.abs(Zxx))

    t_size = -(-t_seconds * fs) // hop_size
    f_freqs = -(-f_freqs * fft_length) // fs
    plt.pcolormesh(t[:t_size], f[:f_freqs], Zxx[:f_freqs, :t_size], cmap='binary', shading='gouraud')
    plt.title('Spectral Transform')
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    plt.savefig(filename, dpi=128, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='./output/spectral')
    args = parser.parse_args()

    fs, audio = load_wav_file(args.input, merge_channels=True)

    f, t, Zxx = spectral_transform(audio, fs)
    plot_spectral_transform(f, t, Zxx, fs, args.output)

if __name__ == '__main__':
    main()
