import argparse

from spectral import hop_size
from utils import load_melody

def evaluate_melody(melody, melody_hop_size, reference_file, reference_hop_size, fs):
    assert melody_hop_size <= reference_hop_size

    # metrics counters
    recall = 0
    false_alarm = 0
    pitch_accuracy = 0
    overall_accuracy = 0
    total_voiced = 0
    total_unvoiced = 0
    total = 0

    melody_index = 0
    with open(reference_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            _, ref_freq = line.strip().split()
            ref_freq = float(ref_freq)

            ref_time = i * (reference_hop_size / fs)
            melody_start_time = melody_index * (melody_hop_size / fs)
            melody_end_time = (melody_index + 1) * (melody_hop_size / fs)
            while not melody_start_time <= ref_time <= melody_end_time:
                melody_index += 1
                melody_start_time = melody_index * (melody_hop_size / fs)
                melody_end_time = (melody_index + 1) * (melody_hop_size / fs)

            if melody_index == len(melody) - 1 or ref_time - melody_start_time <= melody_end_time - ref_time:
                # closer to start frame in melody
                index = melody_index
            else:
                # closer to end frame in melody
                index = melody_index + 1

            # recall / false alarm
            if melody[index] != 0 and ref_freq != 0:
                recall += 1
            elif melody[index] != 0 and ref_freq == 0:
                false_alarm += 1

            # accuracy / pitch accuracy
            if ref_freq * (2**(-0.5 / 12)) <= melody[index] <= ref_freq * (2**(0.5 / 12)):
                pitch_accuracy += 1
            if (melody[index] == 0 and ref_freq == 0) or (melody[index] != 0 and ref_freq != 0 and \
                    ref_freq * (2**(-0.5 / 12)) <= melody[index] <= ref_freq * (2**(0.5 / 12))):
                overall_accuracy += 1

            # total voiced / total unvoiced / total
            if ref_freq != 0:
                total_voiced += 1
            else:
                total_unvoiced += 1
            total += 1

    print(f'voiced recall rate: {recall / total_voiced}')
    print(f'voiced false alarm rate: {false_alarm / total_unvoiced}')
    print(f'raw pitch accuracy: {pitch_accuracy / total}')
    print(f'overall accuracy: {overall_accuracy / total}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--reference_hop_size', type=int, required=True)
    args = parser.parse_args()

    melody = load_melody('./output/melody.txt')
    evaluate_melody(melody, hop_size, args.reference, args.reference_hop_size, 44100)

if __name__ == '__main__':
    main()
