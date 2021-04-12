import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

fs = 44100

# filter coefficients from http://replaygain.hydrogenaud.io/proposal/equal_loudness.html
iir_a = [
    1.00000000000000, -3.47845948550071, 6.36317777566148, -8.54751527471874, 9.47693607801280, -8.81498681370155,
    6.85401540936998, -4.39470996079559, 2.19611684890774, -0.75104302451432, 0.13149317958808
]
iir_b = [
    0.05418656406430, -0.02911007808948, -0.00848709379851, -0.00851165645469, -0.00834990904936, 0.02245293253339,
    -0.02596338512915, 0.01624864962975, -0.00240879051584, 0.00674613682247, -0.00187763777362
]
# butter_a = [1.00000000000000, -1.96977855582618, 0.97022847566350]
# butter_b = [0.98500175787242, -1.97000351574484, 0.98500175787242]

iir_sos = signal.tf2sos(iir_b, iir_a)
butter_sos = signal.butter(2, 150 / (fs / 2), btype='high', output='sos')

def equal_loudness_filter(audio):
    # apply cascaded 10th-order IIR filter and 2nd-order high-pass butterworth filter
    return signal.sosfilt(butter_sos, signal.sosfilt(iir_sos, audio))

def plot_filter_response(b, a, title, output_file):
    w, h = signal.freqz(b, a, fs=fs)
    plt.clf()
    plt.semilogx(w[1:], 20 * np.log10(np.abs(h[1:])))
    plt.title(title)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude (dB)')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig(output_file, dpi=128, bbox_inches='tight')

if __name__ == '__main__':
    # plot frequency response of individual filters
    iir_b, iir_a = signal.sos2tf(iir_sos)
    plot_filter_response(iir_b, iir_a, '10th-order IIR Filter Frequency Response', 'output/iir_filter.png')
    butter_b, butter_a = signal.sos2tf(butter_sos)
    plot_filter_response(butter_b, butter_a, 'Butterworth Filter Frequency Response', 'output/butterworth_filter.png')

    # plot frequency response of cascaded filters
    plot_filter_response(signal.convolve(iir_b, butter_b), signal.convolve(iir_a, butter_a),
                         '10th-order IIR Filter + Butterworth Filter Frequency Response',
                         'output/equal_loudness_filter.png')
