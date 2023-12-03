import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten

import utils.calc_mem
from preprocess import preprocess, WINDOW_SIZE, SPECTRUM_SIZE, SPECTRUM_MEAN, SPECTRUM_STD, SAMPLE_RATE, FRAME_SIZE, FRAME_STRIDE
from utils.export_tflite import write_model_h_file, write_model_c_file
from utils.plots import plot_dataset, plot_learning_curves, plot_convolution_filters, plot_first_positive_window, plot_predictions_vs_labels

# Minimize TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Load and preprocess data
data_dir = '../Data/'
x_files = []
y_files = []
for c_file in os.listdir('../Data/'):
    if c_file.startswith('audio_') and c_file.endswith('.wav'):
        recording_id = c_file[6:-4]
        label_file = 'labels_' + recording_id + '.txt'
        print('Preprocessing ' + c_file + ' and ' + label_file)
        x_file, y_file = preprocess(data_dir + c_file, data_dir + label_file)
        x_files.append(x_file)
        y_files.append(y_file)

# Concatenate files into feature and label arrays
x = np.concatenate(x_files)  # Shape: (number of windows, WINDOW_SIZE, SPECTRUM_SIZE) = (number of windows, 32, 48)
y = np.concatenate(y_files)  # Shape: (number of windows, 1)

# Plot the spectrogram of the entire dataset with labels underneath
plot_dataset(x, y, block=True)

# Split into training, validation and test sets
x_train, x_val, x_test = np.split(x, [int(.6 * len(x)), int(.8 * len(x))])
y_train, y_val, y_test = np.split(y, [int(.6 * len(y)), int(.8 * len(y))])

# Determine class weights
num_positives = np.sum(y)
num_negatives = len(y) - num_positives
ratio = num_negatives / num_positives
class_weights = {0: 1 / np.sqrt(ratio), 1: np.sqrt(ratio)}  # Divide by sqrt(ratio) to make losses comparable
print('Class weights:', class_weights)

# Build and compile model
print('Building model...')
model = Sequential()
model.add(Conv1D(8, 3, activation='relu', input_shape=(WINDOW_SIZE, SPECTRUM_SIZE)))  # Output shape (30, 8)
model.add(Conv1D(8, 3, activation='relu'))  # Output shape (28, 8)
model.add(Flatten())  # Output shape (224,)
model.add(Dense(1, activation='sigmoid'))  # Output shape (1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train model with early stopping and class weights; save best model
print('Training model...')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=16)
model_checkpoint = keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_val, y_val), class_weight=class_weights,
          callbacks=[early_stopping, model_checkpoint])

# Plot learning curves
plot_learning_curves(model, block=False)

# Plot the 8 convolution filters of the first layer
plot_convolution_filters(8, model.layers[0], block=False)

# Plot the first window that has a positive label
plot_first_positive_window(x, y, block=False)

# Load best model
model = keras.models.load_model('model.h5')

# Evaluate model on validation and test sets
val_loss, val_accuracy = model.evaluate(x_val, y_val)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

# Print evaluation metrics
print()
print('Validation loss:     %.4f' % val_loss)
print('Validation accuracy: %.4f' % val_accuracy)
print('Test loss:           %.4f' % test_loss)
print('Test accuracy:       %.4f' % test_accuracy)

# Print confusion matrix
y_pred_bool = np.round(y_pred)
confusion_matrix = np.zeros((2, 2))
for i in range(len(y_pred_bool)):
    confusion_matrix[int(y_test[i]), int(y_pred_bool[i, 0])] += 1
print('True positives:     ', int(confusion_matrix[1, 1]))
print('True negatives:     ', int(confusion_matrix[0, 0]))
print('False positives:    ', int(confusion_matrix[0, 1]))
print('False negatives:    ', int(confusion_matrix[1, 0]))

# Plot predictions vs labels
plot_predictions_vs_labels(y_pred, y_test, block=True)

# Convert to TensorFlow Lite model with quantization
print('Converting to TensorFlow Lite model...')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Export TensorFlow Lite model to C source files
print('Exporting TensorFlow Lite model to C source files...')
defines = {"SAMPLE_RATE": SAMPLE_RATE,
           "FRAME_SIZE": FRAME_SIZE,
           "FRAME_STRIDE": FRAME_STRIDE,
           "WINDOW_SIZE": WINDOW_SIZE,
           "SPECTRUM_SIZE": SPECTRUM_SIZE,
           "SPECTRUM_MEAN": SPECTRUM_MEAN,
           "SPECTRUM_STD": SPECTRUM_STD}
write_model_h_file("../ESP-32/main/model.h", defines)
write_model_c_file('../ESP-32/main/model.c', tflite_model)

# Save TensorFlow Lite model and print memory
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
utils.calc_mem.calc_mem("model.tflite")

print('Done.')
