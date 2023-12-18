"""
Test Tfite model
================

It's also possible to test the .tflite model in python enviroment before deploying it to the microcontroller.


"""


import numpy as np
import tensorflow as tf


model = "gen/model_quantized.tflite"


# Load x_test from main.py
x_test = np.load("gen/x_test.npy")
y_test = np.load("gen/y_test.npy")
y_pred_quantized = np.empty((4339, 1, 1), dtype=np.int8)

# Load interpreter
interpreter = tf.lite.Interpreter(model_path=model)
# Allocate tensors
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Get scaling values for quantized input
input_scale, input_zero_point = input_details[0]["quantization"]
print(f"Input scale: {input_scale}, Input zero point: {input_zero_point}")
# Quantize x_test
x_test_quantized = x_test / input_scale + input_zero_point
# Convert to uint8
x_test_quantized_int = x_test_quantized.astype(np.int8)
np.save("x_test_quantized.npy", x_test_quantized[1])
for i in range(len(x_test_quantized_int)):
    interpreter.set_tensor(
        input_details[0]["index"], x_test_quantized_int[i].reshape(1, 24, 28)
    )
    interpreter.invoke()
    # Get output tensor
    y_pred_quantized[1] = interpreter.get_tensor(output_details[0]["index"])

# Get output scales to dequantize output
output_scale, output_zero_point = output_details[0]["quantization"]
# Dequantize output
y_pred = y_pred_quantized.astype(np.float32)
output_f32 = (y_pred - output_zero_point) * output_scale
# Print evaluation metrics
output_f32 = np.round(output_f32)
confusion_matrix = np.zeros((2, 2))
for i in range(len(output_f32)):
    confusion_matrix[int(y_test[i]), int(output_f32[i, 0])] += 1
print("True positives:     ", int(confusion_matrix[1, 1]))
print("True negatives:     ", int(confusion_matrix[0, 0]))
print("False positives:    ", int(confusion_matrix[0, 1]))
print("False negatives:    ", int(confusion_matrix[1, 0]))
