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

    reference = load_melody(reference_file)

    for i, melody_freq in enumerate(melody):
        melody_time = i * (melody_hop_size / fs)

        # get index of reference at the corresponding time of melody_time
        ref_index = min(round(melody_time / (reference_hop_size / fs)), len(reference) - 1)
        ref_freq = reference[ref_index]

        # recall / false alarm
        if melody_freq != 0 and ref_freq != 0:
            recall += 1
        elif melody_freq != 0 and ref_freq == 0:
            false_alarm += 1

        # accuracy / pitch accuracy
        if ref_freq * (2**(-0.5 / 12)) <= melody_freq <= ref_freq * (2**(0.5 / 12)):
            pitch_accuracy += 1
        if (melody_freq == 0 and ref_freq == 0) or (melody_freq != 0 and ref_freq != 0 and \
                ref_freq * (2**(-0.5 / 12)) <= melody_freq <= ref_freq * (2**(0.5 / 12))):
            overall_accuracy += 1

        # total voiced / total unvoiced / total
        if ref_freq != 0:
            total_voiced += 1
        else:
            total_unvoiced += 1
        total += 1

    voiced_recall = recall / total_voiced
    print(f'voiced recall rate: {voiced_recall:.3f} ({recall} of {total_voiced})')
    voiced_false_alarm = false_alarm / total_unvoiced
    print(f'voiced false alarm rate: {voiced_false_alarm:.3f} ({false_alarm} of {total_unvoiced})')
    raw_pitch_acc = pitch_accuracy / total
    print(f'raw pitch accuracy: {raw_pitch_acc:.3f} ({pitch_accuracy} of {total})')
    overall_acc = overall_accuracy / total
    print(f'overall accuracy: {overall_acc:.3f} ({overall_accuracy} of {total})')

    return voiced_recall, voiced_false_alarm, raw_pitch_acc, overall_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--reference_hop_size', type=int, required=True)
    args = parser.parse_args()

    melody = load_melody('./output/melody.txt')
    evaluate_melody(melody, hop_size, args.reference, args.reference_hop_size, 44100)

if __name__ == '__main__':
    main()
