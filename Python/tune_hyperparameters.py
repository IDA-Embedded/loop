import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from preprocess import preprocess_all, WINDOW_SIZE, SPECTRUM_SIZE
from utils.plots import plot_hyperparameter_tuning_results

# Minimize TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def tune_hyperparameters():
    # Preprocess data if not done already
    if not os.path.exists('x.npy') or not os.path.exists('y.npy'):
        preprocess_all('../Data/')

    # Load preprocessed data
    x = np.load('x.npy')
    y = np.load('y.npy')

    # Split into training, validation and test sets
    x_train, x_val, _ = np.split(x, [int(.6 * len(x)), int(.8 * len(x))])
    y_train, y_val, _ = np.split(y, [int(.6 * len(y)), int(.8 * len(y))])

    # Shuffle the training data
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    # Define hyperparameter search space
    dropout_rates = [0.0, 0.1, 0.2]
    num_filters = [8, 16, 32]
    kernel_sizes = [3, 5, 7]
    learning_rates = [0.0020]  # [0.0002, 0.0005, 0.0010, 0.0020]
    batch_sizes = [64]  # [64, 128, 256]
    attempt_count = 3  # 5

    # Perform grid search
    average_grid = np.zeros((len(dropout_rates), len(num_filters), len(kernel_sizes), len(learning_rates), len(batch_sizes)))
    for dropout_rate in dropout_rates:
        for num_filter in num_filters:
            for kernel_size in kernel_sizes:
                for learning_rate in learning_rates:
                    for batch_size in batch_sizes:
                        # Print current hyperparameters
                        print('Dropout rate {0}, num filters {1:2d}, kernel size {2}, learning rate {3:.4f}, batch size {4:3d}...'.format(
                            dropout_rate, num_filter, kernel_size, learning_rate, batch_size), end='')

                        # Train model multiple times and calculate average validation loss
                        lowest = 1.0
                        average = 0.0
                        for i in range(attempt_count):
                            result = _train(x_train, y_train, x_val, y_val, dropout_rate, num_filter, kernel_size, learning_rate, batch_size)
                            print(' {0:.6f}'.format(result), end='')
                            if result < lowest:
                                lowest = result
                            average += result / attempt_count
                        print(' (lowest: {0:.6f}, average: {1:.6f})'.format(lowest, average))

                        # Store average validation loss in grid
                        average_grid[dropout_rates.index(dropout_rate), num_filters.index(num_filter), kernel_sizes.index(kernel_size), learning_rates.index(learning_rate), batch_sizes.index(batch_size)] = average

    # Find the best combination of hyperparameters based on average validation loss
    best = np.unravel_index(np.argmin(average_grid), average_grid.shape)
    print('Best combination of hyperparameters (validation loss {:.6f}):'.format(average_grid[best]))
    print('Dropout rate:  ', dropout_rates[best[0]])
    print('Num filters:   ', num_filters[best[1]])
    print('Kernel size:   ', kernel_sizes[best[2]])
    print('Learning rate: ', learning_rates[best[3]])
    print('Batch size:    ', batch_sizes[best[4]])

    # Reduce average grid to 3D array before plotting - assume fixed learning rate and batch size
    average_grid = average_grid[:, :, :, learning_rates.index(0.0020), batch_sizes.index(64)]

    # Plot results
    title = 'Average validation loss for learning rate 0.002 and batch size 64'
    plot_hyperparameter_tuning_results(title, average_grid, dropout_rates, num_filters, kernel_sizes, block=True)


def _train(x_train, y_train, x_val, y_val, dropout_rate, num_filters, kernel_size, learning_rate, batch_size):
    # Build and compile model
    model = Sequential()
    model.add(Conv1D(num_filters, kernel_size, activation='relu', input_shape=(WINDOW_SIZE, SPECTRUM_SIZE)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy')

    # Train model with early stopping and class weights; save best model
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
    model_checkpoint = keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
    model.fit(x_train, y_train, epochs=100, batch_size=batch_size, validation_data=(x_val, y_val),
              callbacks=[early_stopping, model_checkpoint], verbose=0)

    # Load best model
    model = keras.models.load_model('model.h5')

    # Evaluate model on validation and test sets
    return model.evaluate(x_val, y_val, verbose=0)


if __name__ == '__main__':
    tune_hyperparameters()
