import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

fs = 44100

# filter coefficients from http://replaygain.hydrogenaud.io/proposal/equal_loudness.html
yw_a = [
    1.00000000000000, -3.47845948550071, 6.36317777566148, -8.54751527471874, 9.47693607801280, -8.81498681370155,
    6.85401540936998, -4.39470996079559, 2.19611684890774, -0.75104302451432, 0.13149317958808
]
yw_b = [
    0.05418656406430, -0.02911007808948, -0.00848709379851, -0.00851165645469, -0.00834990904936, 0.02245293253339,
    -0.02596338512915, 0.01624864962975, -0.00240879051584, 0.00674613682247, -0.00187763777362
]
# butter_a = [1.00000000000000, -1.96977855582618, 0.97022847566350]
# butter_b = [0.98500175787242, -1.97000351574484, 0.98500175787242]

yw_sos = signal.tf2sos(yw_b, yw_a)
butter_sos = signal.butter(2, 150 / (fs / 2), btype='high', output='sos')

def equal_loudness_filter(audio):
    # apply cascaded 10th-order IIR filter and 2nd-order high-pass butterworth filter
    return signal.sosfilt(butter_sos, signal.sosfilt(yw_sos, audio))

def plot_filter_response(filter, output_file):
    w, h = signal.sosfreqz(filter, fs=fs)
    plt.semilogx(w, 20 * np.log10(np.abs(h)))
    plt.xlabel('Frequency (hz)')
    plt.ylabel('Amplitude (dB)')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig(output_file, dpi=240)

if __name__ == '__main__':
    # plot frequency response of filters
    plot_filter_response(yw_sos, 'output/yw_filter.png')
    plot_filter_response(butter_sos, 'output/butterworth_filter.png')
