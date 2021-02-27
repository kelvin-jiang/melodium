import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, stft

from utils import load_wav_file

M = 2048  # window length
H = 128  # hop size
N = 8192  # FFT length

def spectral_transform(audio, sampling_rate):
    f, t, Zxx = stft(audio, sampling_rate, nperseg=M, noverlap=M - H, nfft=N)
    return f, t, Zxx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_audio_file', required=True)
    parser.add_argument('--output_spectrogram_file', required=True)
    args = parser.parse_args()

    sampling_rate, audio = load_wav_file(args.input_audio_file, merge_channels=True)

    # STFT
    f, t, Zxx = spectral_transform(audio, sampling_rate)

    # peak finding
    for i in range(Zxx.shape[1]):
        # TODO: tune parameters for find_peaks
        peaks, _ = find_peaks(np.abs(Zxx[:, i]))
        non_peaks = np.setdiff1d(np.arange(Zxx.shape[0]), peaks)
        np.put(Zxx[:, i], non_peaks, 0)

    plt.pcolormesh(t, f[:400], np.abs(Zxx[:400]), cmap='Reds', shading='gouraud')
    plt.savefig(args.output_spectrogram_file, dpi=1024)
