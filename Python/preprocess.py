from math import ceil, floor

import numpy as np
import pandas as pd
from scipy.io import wavfile

# Sample rate is 16kHz
SAMPLE_RATE = 16000

# FFT frame size is 256 samples (16ms) and stride is 128 samples (8ms)
FRAME_SIZE = 256
FRAME_STRIDE = 256

# Window size is 32 spectral frames (256ms) and stride is 8 frames (64ms)
# We will not run inference at this high rate, but it's a good way to get more training data
WINDOW_SIZE = 32
WINDOW_STRIDE = 8
SPECTRUM_SIZE = FRAME_SIZE * 3000 // SAMPLE_RATE   # Only keep frequency bins up to 3000 Hz

# Mean and std of the spectrogram of all frames for normalization
SPECTRUM_MEAN = 9.0
SPECTRUM_STD = 1.2


def preprocess(wav_file: str, label_file: str):
    # Load wav file
    sample_rate, sound_data = wavfile.read(wav_file)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f'Expected sample rate of {SAMPLE_RATE}, but got {sample_rate}')

    # Preprocess data with hamming window and fourier transform
    spectral_frames = []
    for j in range(0, len(sound_data) - FRAME_SIZE, FRAME_STRIDE):
        frame = sound_data[j:j + FRAME_SIZE]
        frame = frame - np.average(frame)
        frame = frame * np.hamming(FRAME_SIZE)
        spectral_frame = np.abs(np.fft.rfft(frame))
        spectral_frame = spectral_frame[0:SPECTRUM_SIZE]
        spectral_frame = np.log1p(np.abs(spectral_frame))
        spectral_frames.append(spectral_frame)

    # Convert to numpy array
    spectral_frames = np.array(spectral_frames)

    # Normalize data
    spectral_frames = (spectral_frames - SPECTRUM_MEAN) / SPECTRUM_STD

    # Stack frames into windows
    windows = []
    for i in range(0, len(spectral_frames) - WINDOW_SIZE, WINDOW_STRIDE):
        window = spectral_frames[i:i + WINDOW_SIZE]
        windows.append(window)

    # Convert to numpy array
    x = np.array(windows)

    # Load labels
    labels = pd.read_csv(label_file, sep='\t', header=None, names=['start', 'end', 'label'])

    # Convert time range labels to window labels
    # We want to label windows that are completely within the time range of a label as positive,
    # and all other windows as negative.
    # Example:
    # Label:              ****************
    # Start/end:           |           |
    # Windows: | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
    y = np.zeros(len(windows))
    windows_per_second = sample_rate / FRAME_STRIDE / WINDOW_STRIDE  # = 1 / 0.064s = 15.625 Hz
    for index, label in labels.iterrows():
        start_window = ceil(label['start'] * windows_per_second)
        end_window = floor(label['end'] * windows_per_second)
        y[start_window:end_window] = 1

    return x, y
