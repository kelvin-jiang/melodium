import numpy as np
from scipy.io import wavfile as wav

def load_wav_file(audio_file, merge_channels=False):
    sampling_rate, audio = wav.read(audio_file)
    if merge_channels and len(audio.shape) > 1:
        original_dtype = audio.dtype
        audio = np.mean(audio, axis=1).astype(original_dtype)  # cast back to original datatype
    return sampling_rate, audio

def write_wav_file(melody, melody_file, sampling_rate):
    wav.write(melody_file, sampling_rate, melody)
