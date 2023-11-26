#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Include TFLM
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "examples/micro_speech/main/feature_provider.h"

// Project includes
#include "audio.h"
#include "preprocess.h"
#include "model.h"

// Compile time check preprocessor constants
#if SAMPLE_RATE != AUDIO_SAMPLE_RATE
#error "AUDIO_SAMPLE_RATE must be equal to SAMPLE_RATE from model.h"
#endif
#if FRAME_SIZE != AUDIO_BUFFER_SAMPLES
#error "AUDIO_BUFFER_SAMPLES must be equal to FRAME_SIZE from model.h"
#endif

// Tensor arena size, found by trial and error
#define TENSOR_ARENA_SIZE (30 * 1024)

// Static variables
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

/**
 * @brief Main setup function.
 */
void setup(void)
{
    // Load TFlite model
    model = tflite::GetModel(model_binary);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        printf("Model schema mismatch!\n");
        abort();
    }

    // Create an interpreter
    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    // Conv1D
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddConv2D();
    // Flatten
    micro_op_resolver.AddExpandDims();
    // Dense with sigmoid activation
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddLogistic();
    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    // Allocate memory for input and output tensors
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        printf("Failed to allocate tensors!");
        abort();
    }

    // Get pointers for input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Print input and output tensor dimensions
    printf("Input tensor shape: %d, %d, %d\n", input->dims->data[0], input->dims->data[1], input->dims->data[2]);
    printf("Output tensor shape: %d\n", output->dims->data[0]);
    
    // Initialize audio with gain 16.0 (found experimentally)
    audio_init(16.0f, FRAME_STRIDE);

    // Initialize preprocessing
    if (!preprocess_init())
    {
        printf("Failed to initialize preprocessing!\n");
        abort();
    }
}

/**
 * @brief Main loop function.
 */
void loop(void)
{
    // Obtain audio frame
    float* audio_frame = audio_read();

    // Put and preprocess audio frame
    preprocess_put_audio(audio_frame);

    // Obtain features and run inference if features are ready
    float amplitude;
    if (preprocess_get_features(input->data.f, &amplitude))
    {
        // Run inference
        if (interpreter->Invoke() != kTfLiteOk)
            printf("Failed to invoke interpreter!\n");

        // Print output
        printf("Amplitude: %5.0f, Prediction: %.2f\n", amplitude, output->data.f[0]);
    }
}

extern "C" void app_main(void)
{
    setup();
    while (true) {
        loop();
    }
}
