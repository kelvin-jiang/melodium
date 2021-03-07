import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, stft

from utils import load_wav_file

M = 2048  # window length
H = 128  # hop size
N = 8192  # FFT length

def spectral_transform(audio, fs):
    f, t, Zxx = stft(audio, fs, nperseg=M, noverlap=M - H, nfft=N)

    return f, t, Zxx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_audio_file', required=True)
    parser.add_argument('--t_seconds', type=int, default=10)
    parser.add_argument('--f_freqs', type=int, default=2000)
    parser.add_argument('--output_spectrogram_file', required=True)
    args = parser.parse_args()

    fs, audio = load_wav_file(args.input_audio_file, merge_channels=True)

    f, t, Zxx = spectral_transform(audio, fs)

    # peak finding (for each time window)
    for i in range(Zxx.shape[1]):
        peaks, _ = find_peaks(np.abs(Zxx[:, i]))
        non_peaks = np.setdiff1d(np.arange(Zxx.shape[0]), peaks)
        np.put(Zxx[:, i], non_peaks, 0)

    # apply sign function on magnitudes to make peaks visible
    Zxx = np.sign(np.abs(Zxx))

    t_size = -(-args.t_seconds * fs) // H
    f_freqs = -(-args.f_freqs * N) // fs

    plt.pcolormesh(t[:t_size], f[:f_freqs], Zxx[:f_freqs, :t_size], cmap='binary', shading='gouraud')
    plt.savefig(args.output_spectrogram_file, dpi=128)

if __name__ == '__main__':
    main()
