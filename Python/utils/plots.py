from matplotlib import pyplot as plt
import keras

from preprocess import *


def plot_dataset(x: np.ndarray, y: np.ndarray, block: bool):
    # Concatenate the windows along the time axis
    # However, there are overlapping windows, so we should only use the first WINDOW_STRIDE frames of each window and skip the rest.
    x_concatenated = np.concatenate(x[:, 0:WINDOW_STRIDE, :], axis=0)

    # Limit values to [-2, 2] for better visualization
    x_concatenated = np.clip(x_concatenated, -2, 2)

    # Create plot and clear axes
    plt.figure(figsize=(12, 6))
    plt.title('Spectrogram of entire data set')
    plt.xticks([])
    plt.yticks([])

    # Plot the spectrogram
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.imshow(x_concatenated.T, aspect='auto', origin='lower', interpolation='none')
    ax1.set_ylabel('Frequency bin')
    ax1.set_xticks([])
    tick_frames = np.arange(0, len(x_concatenated), 60 * 5 * SAMPLE_RATE / FRAME_STRIDE)
    tick_labels = np.round(tick_frames * FRAME_STRIDE / SAMPLE_RATE / 60, 2)
    ax1.set_xticks(tick_frames, labels=tick_labels)
    ax1.set_xticklabels(tick_labels)

    # Plot the labels
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax2.plot(y)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Label')
    tick_frames = np.arange(0, len(y), 60 * 5 * SAMPLE_RATE / FRAME_STRIDE / WINDOW_STRIDE)
    tick_labels = np.round(tick_frames * WINDOW_STRIDE * FRAME_STRIDE / SAMPLE_RATE / 60, 2)
    ax2.set_xticks(tick_frames)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlim(0, len(y) - 1)
    ax2.set_yticks([])

    # Show plot
    plt.tight_layout()
    plt.show(block=block)


def plot_learning_curves(model: keras.models.Model, block: bool):
    plt.figure(figsize=(12, 6))
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('Learning curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training loss', 'Validation loss'])
    plt.show(block=block)


def plot_convolution_filters(filters: int, layer: keras.layers.Conv1D, block: bool):
    plt.figure(figsize=(6, 6))
    for i in range(filters):
        plt.subplot(1, filters, i + 1)
        plt.imshow(layer.get_weights()[0][:, :, i].T, aspect='auto', origin='lower', interpolation='none')
        plt.title('Filter ' + str(i))
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('Convolution filters of first layer')
    plt.show(block=block)


def plot_first_positive_window(x: np.ndarray, y: np.ndarray, block: bool):
    plt.figure(figsize=(6, 6))
    for i in range(len(y)):
        if y[i] == 1:
            plt.imshow(x[i, :, :].T, aspect='auto', origin='lower', interpolation='none')
            plt.title('First window with positive label')
            plt.xlabel('Frame')
            plt.ylabel('Frequency bin')
            plt.xticks([])
            plt.yticks([])
            break
    plt.show(block=block)


def plot_predictions_vs_labels(y_pred: np.ndarray, y_test: np.ndarray, block: bool):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test)
    plt.plot(-y_pred)  # Negate predictions to make them easier to see
    plt.title('Test set predictions vs labels')
    plt.xlabel('Frame')
    plt.ylabel('Prediction')
    plt.legend(['Label', 'Prediction'])
    plt.show(block=block)
