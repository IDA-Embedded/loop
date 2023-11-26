import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from matplotlib import pyplot as plt

from preprocess import *

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
x = np.concatenate(x_files)
y = np.concatenate(y_files)

# Shuffle data, but only what will become training and validation sets
# We will keep the test set as a contiguous sequence for plotting
indices = np.arange(int(.8 * len(x)))
np.random.shuffle(indices)
indices = np.concatenate((indices, np.arange(int(.8 * len(x)), len(x))))
x = x[indices]
y = y[indices]

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
model.add(Conv1D(8, 3, activation='relu', input_shape=(WINDOW_SIZE, SPECTRUM_SIZE)))
model.add(Conv1D(8, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model with early stopping and class weights; save best model
print('Training model...')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=16)
model_checkpoint = keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_val, y_val), class_weight=class_weights,
          callbacks=[early_stopping, model_checkpoint])

# Plot learning curves
plt.figure(figsize=(12, 6))
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Learning curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training loss', 'Validation loss'])
plt.show()

# Load best model
model = keras.models.load_model('model.h5')

# Evaluate model with confusion matrix
loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)
confusion_matrix = np.zeros((2, 2))
for i in range(len(y_pred)):
    confusion_matrix[int(y_test[i]), int(y_pred[i, 0])] += 1
print('Correct positives:', int(confusion_matrix[1, 1]))
print('Correct negatives:', int(confusion_matrix[0, 0]))
print('False positives:', int(confusion_matrix[0, 1]))
print('False negatives:', int(confusion_matrix[1, 0]))

# Plot y_pred vs y_test
# Put them in different subplots to make them easier to compare
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(y_test)
plt.title('y_test')
plt.xlabel('Window')
plt.ylabel('Label')
plt.subplot(2, 1, 2)
plt.plot(y_pred)
plt.title('y_pred')
plt.xlabel('Window')
plt.ylabel('Label')
plt.show()

# Convert to TensorFlow Lite model with quantization
print('Converting to TensorFlow Lite model...')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Export TensorFlow Lite model to c++ source file
print('Exporting TensorFlow Lite model to c++ source files...')
with open("../ESP-32/main/model.h", 'w') as h_file:
    h_file.write('#ifndef MODEL_H\n')
    h_file.write('#define MODEL_H\n')
    h_file.write('\n')
    h_file.write('#define SAMPLE_RATE ' + str(SAMPLE_RATE) + '\n')
    h_file.write('#define FRAME_SIZE ' + str(FRAME_SIZE) + '\n')
    h_file.write('#define FRAME_STRIDE ' + str(FRAME_STRIDE) + '\n')
    h_file.write('#define WINDOW_SIZE ' + str(WINDOW_SIZE) + '\n')
    h_file.write('#define SPECTRUM_SIZE ' + str(SPECTRUM_SIZE) + '\n')
    h_file.write('#define SPECTRUM_MEAN ' + str(SPECTRUM_MEAN) + '\n')
    h_file.write('#define SPECTRUM_STD ' + str(SPECTRUM_STD) + '\n')
    h_file.write('\n')
    h_file.write('extern const unsigned char model_binary[];\n')
    h_file.write('\n')
    h_file.write('#endif\n')

with open("../ESP-32/main/model.c", 'w') as c_file:
    c_file.write('const unsigned char model_binary[] = {\n')
    for i, byte in enumerate(tflite_model):
        c_file.write(f'0x{byte:02x}, ')
        if (i + 1) % 12 == 0:
            c_file.write('\n')
    c_file.write('\n};\n')

print('Done.')
