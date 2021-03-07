import numpy as np
from scipy.io import wavfile as wav

def load_wav_file(audio_file, merge_channels=False):
    fs, audio = wav.read(audio_file)
    if merge_channels and len(audio.shape) > 1:
        original_dtype = audio.dtype
        audio = np.mean(audio, axis=1).astype(original_dtype)  # cast back to original datatype
    return fs, audio

def write_wav_file(melody, melody_file, fs):
    wav.write(melody_file, fs, melody)
