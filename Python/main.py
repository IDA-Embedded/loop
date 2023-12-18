import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout

from preprocess import preprocess_all, WINDOW_SIZE, SPECTRUM_SIZE, SPECTRUM_TOP, SPECTRUM_SRC, SPECTRUM_DST, SPECTRUM_MEAN, SPECTRUM_STD, SAMPLE_RATE, FRAME_SIZE, FRAME_STRIDE
from utils.calc_mem import calc_mem
from utils.export_tflite import write_model_h_file, write_model_c_file
from utils.plots import plot_predictions_vs_labels, plot_learning_curves

# Enable quantization
ENABLE_QUANTIZATION = False

# Minimize TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# Preprocess data if not done already
if not os.path.exists('gen/x.npy') or not os.path.exists('gen/y.npy'):
    preprocess_all('../Data/')

# Load preprocessed data
x = np.load('gen/x.npy')
y = np.load('gen/y.npy')

# Plot the spectrogram of the entire dataset with labels underneath
# plot_dataset(x, y, block=True)

# Split into training, validation and test sets
x_train, x_val, x_test = np.split(x, [int(.6 * len(x)), int(.8 * len(x))])
y_train, y_val, y_test = np.split(y, [int(.6 * len(y)), int(.8 * len(y))])

# Save x/y test for tflite test
np.save("gen/x_test.npy", x_test)
np.save("gen/y_test.npy", y_test)

# Shuffle the training data
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

# Determine negative to positive ratio
num_positives = np.sum(y)
num_negatives = len(y) - num_positives
ratio = num_negatives / num_positives
print('Negative to positive ratio: ', ratio)

# Build and compile model
print('Building model...')
model = Sequential()
model.add(Conv1D(8, 3, activation='relu', input_shape=(WINDOW_SIZE, SPECTRUM_SIZE)))  # Output shape (22, 8)
model.add(MaxPooling1D(2))  # Output shape (11, 8)
model.add(Dropout(0.2))
model.add(Conv1D(8, 3, activation='relu'))  # Output shape (9, 8)
model.add(MaxPooling1D(2))  # Output shape (4, 8)
model.add(Dropout(0.2))
model.add(Flatten())  # Output shape (32)
model.add(Dense(1, activation='sigmoid'))  # Output shape (1)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train model with early stopping; save best model
print('Training model...')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=16)
model_checkpoint = keras.callbacks.ModelCheckpoint('gen/model.h5', monitor='val_loss', save_best_only=True)
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val),
          callbacks=[early_stopping, model_checkpoint])

# Plot learning curves
plot_learning_curves(model, block=False)

# Plot the first window that has a positive label
# plot_first_positive_window(x, y, block=False)

# Plot the 8 convolution filters of the first layer
# plot_convolution_filters(8, model.layers[0], block=False)

# Load best model
model = keras.models.load_model('gen/model.h5')

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

# Convert to TensorFlow Lite model
print('Converting to TensorFlow Lite model...')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
if ENABLE_QUANTIZATION:
    # Function for generating representative data
    def representative_dataset():
        global x_train
        x_train_samples = x_train[np.random.choice(x_train.shape[0], 5000, replace=False)]
        yield [x_train_samples.astype(np.float32)]

    # Quantize model
    print("Quantizing TensorFlow Lite model...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8  # or tf.int8

tflite_model = converter.convert()

# Export TensorFlow Lite model to C source files
print("Exporting TensorFlow Lite model to C source files...")
defines = {
    "SAMPLE_RATE": SAMPLE_RATE,
    "FRAME_SIZE": FRAME_SIZE,
    "FRAME_STRIDE": FRAME_STRIDE,
    "WINDOW_SIZE": WINDOW_SIZE,
    "SPECTRUM_TOP": SPECTRUM_TOP,
    "SPECTRUM_SIZE": SPECTRUM_SIZE,
    "SPECTRUM_MEAN": SPECTRUM_MEAN,
    "SPECTRUM_STD": SPECTRUM_STD
}
declarations = [
    "const unsigned long SPECTRUM_SRC[] = { " + ", ".join(map(str, SPECTRUM_SRC)) + " };",
    "const unsigned long SPECTRUM_DST[] = { " + ", ".join(map(str, SPECTRUM_DST)) + " };"
]
if ENABLE_QUANTIZATION:
    FILE_NAME = "model_quantized"
    # Do not include quantized model into the main project
    write_model_h_file(f"../esp32_metric/components/models/include_v2/model_v2_quan/{FILE_NAME}_v2.h", defines, declarations)
    write_model_c_file(f"../esp32_metric/components/models/{FILE_NAME}_v2.c", tflite_model)
else:
    FILE_NAME = "model"
    write_model_h_file("../ESP-32/main/model.h", defines, declarations)
    write_model_c_file("../ESP-32/main/model.c", tflite_model)
    write_model_h_file(f"../esp32_metric/components/models/include_v2/model_v2/{FILE_NAME}_v2.h", defines, declarations)
    write_model_c_file(f"../esp32_metric/components/models/{FILE_NAME}_v2.c", tflite_model)

# Save TensorFlow Lite model and print memory
with open(f"gen/{FILE_NAME}.tflite", "wb") as f:
    f.write(tflite_model)
calc_mem(f"gen/{FILE_NAME}.tflite")

print("Done.")
