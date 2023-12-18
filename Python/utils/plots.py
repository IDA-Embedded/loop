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
    plt.plot(y_pred, color='blue')
    plt.plot(-y_test, color='green')  # Negate labels to make them easier to see
    plt.title('Test set predictions vs labels')
    plt.xlabel('Window')
    plt.ylabel('Prediction')
    plt.legend(['Prediction', 'Label'])
    plt.show(block=block)


def plot_hyperparameter_tuning_results(title: str, average_grid: np.ndarray, dropout_rates: list, num_filters: list, filter_lengths: list, block: bool):
    # Parameters:
    # average_grid: np.ndarray
    #     3-dimensional array with average validation losses for each combination of dropout rate, num filters and kernel size
    # block: bool
    #     Whether to block the program until the plot is closed

    # Plot all averages as heatmap subplots, where each heatmap shows num_filters vs kernel_size for a specific dropout_rate
    plt.figure(figsize=(12, 4))
    for dropout_rate_index in range(len(dropout_rates)):
        plt.subplot(1, len(dropout_rates), dropout_rate_index + 1)
        # Plot heatmap; use green for 0.0 (min) and red for 0.03 (max)
        plt.imshow(0.03 - average_grid[dropout_rate_index, :, :], cmap='RdYlGn', vmin=0.0, vmax=0.03)
        # Write the value in each cell
        for j in range(len(num_filters)):
            for k in range(len(filter_lengths)):
                plt.text(k, j, '{0:.6f}'.format(average_grid[dropout_rate_index, j, k]), ha='center', va='center', color='black')
        # Set labels and ticks
        plt.title('Dropout rate ' + str(dropout_rates[dropout_rate_index]))
        plt.xlabel('Filter length')
        plt.ylabel('Number of filters')
        plt.xticks(range(len(filter_lengths)), [str(x) for x in filter_lengths])
        plt.yticks(range(len(num_filters)), [str(x) for x in num_filters])

    # Set title
    plt.suptitle(title)

    # Show plot
    plt.show(block=block)
