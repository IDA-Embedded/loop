import os
from math import ceil

import numpy as np
import pandas as pd
from scipy.io import wavfile

# Sample rate is 16kHz
SAMPLE_RATE = 16000

# FFT frame size is 256 samples (16ms) and stride is 256 samples (16ms)
FRAME_SIZE = 256
FRAME_STRIDE = 256

# Window size is 24 spectral frames (384ms) and stride is 4 frames (64ms)
# We will not run inference at this high rate, but it's a good way to get more training data
WINDOW_SIZE = 24
WINDOW_STRIDE = 4

# Bin indexes for reducing spectrum by averaging some bins together into single bins
# The resulting spectral frame has 28 bins as follows:
# 0:     Average of original spectrum bins 0-9
# 1-3:   Copy of original spectrum bins 9-11 (bell undertone)
# 4:     Average of original spectrum bins 12-15
# 5-7:   Copy of original spectrum bins 16-18 (bell base tone)
# 8:     Average of original spectrum bins 19-22
# 9-12:  Copy of original spectrum bins 23-26 (bell overtone 1)
# 13:    Average of original spectrum bins 27-32
# 14-16: Copy of original spectrum bins 33-35 (bell overtone 2)
# 17:    Average of original spectrum bins 36-42
# 18-20: Copy of original spectrum bins 43-45 (bell overtone 3)
# 21:    Average of original spectrum bins 46-52
# 22-24: Copy of original spectrum bins 53-55 (bell overtone 4)
# 25:    Average of original spectrum bins 56-64
# 26-27: Copy of original bins 65-66 (bell overtone 5)
SPECTRUM_SRC = [0, 9, 12, 16, 19, 23, 27, 33, 36, 43, 46, 53, 56, 65, 67]
SPECTRUM_DST = [0, 1,  4,  5,  8,  9, 13, 14, 17, 18, 21, 22, 25, 26, 28]
SPECTRUM_TOP = SPECTRUM_SRC[-1]
SPECTRUM_SIZE = SPECTRUM_DST[-1]

# Mean and std of the spectrogram of all frames for normalization
SPECTRUM_MEAN = 9.0
SPECTRUM_STD = 1.2


def preprocess_all(data_dir: str):
    # Load and preprocess all recordings in the data folder
    x_files = []
    y_files = []
    for c_file in os.listdir('../Data/'):
        if c_file.startswith('audio_') and c_file.endswith('.wav'):
            recording_id = c_file[6:-4]
            label_file = 'labels_' + recording_id + '.txt'
            print('Preprocessing ' + c_file + ' and ' + label_file)
            x_file, y_file = _preprocess_recording(data_dir + c_file, data_dir + label_file)
            x_files.append(x_file)
            y_files.append(y_file)

    # Shuffle files so we get a mix of different recordings in training, validation and test sets
    # Use fixed seed to get reproducible results
    np.random.seed(5)
    indices = np.arange(len(x_files))
    np.random.shuffle(indices)
    x_files_shuffled = []
    y_files_shuffled = []
    for i in indices:
        x_files_shuffled.append(x_files[i])
        y_files_shuffled.append(y_files[i])

    # Concatenate files into feature and label arrays
    x = np.concatenate(x_files_shuffled)  # Shape: (number of windows, WINDOW_SIZE, SPECTRUM_SIZE) = (number of windows, 24, 28)
    y = np.concatenate(y_files_shuffled)  # Shape: (number of windows, 1)

    # Save to files
    np.save('x.npy', x)
    np.save('y.npy', y)


def _preprocess_recording(wav_file: str, label_file: str):
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
        spectral_frame = _reduce_spectrum(spectral_frame)
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
    # and windows that are completely outside the time range as negative. Windows that overlap
    # the start and end time are removed, since it's not clear what label they should have, and
    # we don't want to train the model on ambiguous data.
    #
    # Example:
    # Label:              ************************
    # Windows: |   0   | Remove |   1   |   1   | Remove |   0   |
    #              | Remove |   1   |   1   | Remove |   0   |
    y = np.zeros(len(windows))
    start_or_end_overlap = np.zeros(len(windows), dtype=bool)
    window_period = WINDOW_STRIDE * FRAME_STRIDE / sample_rate  # = 4 * 256 / 16000 = 0.064s
    windows_per_second = 1 / window_period  # = 15.625
    window_length = WINDOW_SIZE * FRAME_STRIDE / sample_rate  # = 24 * 256 / 16000 = 0.384s
    for index, label in labels.iterrows():
        # Compute index of start/end of label in windows
        start_window = ceil(label['start'] * windows_per_second)
        if start_window >= len(y):
            continue
        end_window = ceil((label['end'] - window_length) * windows_per_second)

        # Label all windows within the label range as positive
        y[start_window:end_window] = 1

        # Mark for removal all windows that overlap the start of a range label
        i = start_window - 1
        while i >= 0 and i * window_period + window_length > label['start']:
            start_or_end_overlap[i] = True
            i -= 1

        # Mark for removal all windows that overlap the end of a range label
        i = end_window
        while i < len(y) and i * window_period < label['end']:
            start_or_end_overlap[i] = True
            i += 1

    # Remove windows that overlap the start and end of a range label
    x = x[~start_or_end_overlap]
    y = y[~start_or_end_overlap]

    # If the data is unbalanced, remove some negative windows
    num_positives = np.sum(y)
    num_negatives = len(y) - num_positives
    if int(num_negatives / num_positives) >= 8:
        # Go through the data, and every time we see a negative window, we keep 1 out of ratio negative windows
        # Ratio is chosen such that the data is balanced 4/1, which is acceptable while we're still not throwing
        # away too much data. For example, if there are 50 negative windows for every positive window, then we
        # keep every 12th negative window.
        ratio = int(num_negatives / num_positives) // 4
        keep = np.zeros(len(y), dtype=bool)
        j = 0
        for i in range(len(y)):
            if y[i] == 0:
                if j % ratio == 0:
                    keep[i] = True
                j += 1
            else:
                keep[i] = True
        x = x[keep]
        y = y[keep]

    return x, y


def _reduce_spectrum(spectral_frame: np.ndarray) -> np.ndarray:
    # The incoming spectral frame has 256 bins. We want to keep only the lowest 67 bins,
    # corresponding to everything below 4188 Hz. Moreover, to reduce dimensionality further,
    # we collapse some bins by computing their average into single bins, because they don't
    # contain any bell frequencies, so we don't want to learn any patterns from them.
    reduced_spectrum = np.zeros(SPECTRUM_SIZE)
    for i in range(0, len(SPECTRUM_SRC) - 1, 2):
        # Average the bins that we want to reduce
        reduced_spectrum[SPECTRUM_DST[i]] = np.average(spectral_frame[SPECTRUM_SRC[i]:SPECTRUM_SRC[i + 1]])

        # Copy the bins that we want to keep
        reduced_spectrum[SPECTRUM_DST[i + 1]:SPECTRUM_DST[i + 2]] = spectral_frame[SPECTRUM_SRC[i + 1]:SPECTRUM_SRC[i + 2]]

    return reduced_spectrum


if __name__ == '__main__':
    preprocess_all('../Data/')