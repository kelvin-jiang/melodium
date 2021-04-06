import numpy as np
from scipy.io import wavfile as wav

def load_wav_file(audio_file, merge_channels=False):
    fs, audio = wav.read(audio_file)
    if merge_channels and len(audio.shape) > 1:
        original_dtype = audio.dtype
        audio = np.mean(audio, axis=1).astype(original_dtype)  # cast back to original datatype
    return fs, audio

def write_wav_file(melody, melody_file, fs):
    # normalize to 0 db
    y = melody / np.max(melody)
    wav.write(melody_file, fs, y)

def write_melody(melody, melody_file, fs, hop_size):
    with open(melody_file, 'w', encoding='utf-8') as f:
        for i, curr_freq in enumerate(melody):
            curr_time = i * (hop_size / fs)
            f.write(f'{curr_time: .3f}\t{curr_freq: .3f}\n')
