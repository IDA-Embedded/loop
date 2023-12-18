/**
 * @file metric_model.cpp
 * @author IDA Embedded
 * @brief Very simple example that loads a model and runs inference
 *
 */

// INCLUDES
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_chip_info.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#ifdef CONFIG_MODEL_VERSION_1_0
#include "model_v1.h"
#include "test_data.h"
#elif defined(CONFIG_MODEL_VERSION_2_0) && !defined(CONFIG_USE_QUANTIZED_MODEL)
#include "model_v2.h"
#include "test_data2.h"
#else
#include "model_quantized_v2.h"
#include "test_quantized.h"
#endif

// DEFINES
#define TENSOR_ARENA_SIZE (30 * 1024)

// STATIC VARIABLES
/* Log TAG*/
static const char *TAG_M = "Metric Model";
static tflite::MicroInterpreter *interpreter = nullptr;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
// static gptimer_handle_t gptimer = NULL;

void init_tflu(void)
{
    const tflite::Model *_model = nullptr;

    // Load TFlite model
    _model = tflite::GetModel(model_binary);
    if (_model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG_M, "Model schema mismatch!");
        abort();
    }

#ifdef CONFIG_MODEL_VERSION_1_0
    ESP_LOGI(TAG_M, "Model version 1.0");
    // Create ops resolver - Static or global. Consider during this differently if ever used in production environment.
    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddExpandDims();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddLogistic();
#elif CONFIG_MODEL_VERSION_2_0
    ESP_LOGI(TAG_M, "Model version 2.0");
    static tflite::MicroMutableOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddExpandDims();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddLogistic();
#endif

    // Create interpreter - Static or global. Consider during this differently if ever used in production environment.
    static tflite::MicroInterpreter static_interpreter(_model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    // Allocate memory for input and output tensors
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE(TAG_M, "Failed to allocate tensors!");
        abort();
    }
}

void run_inference(void)
{
    uint64_t total_time = 0;
    // Get input and output tensors
    TfLiteTensor *input = interpreter->input(0);

    for (int i = 0; i < g_x_test_test_data_size; i++)
    {
#ifndef CONFIG_USE_QUANTIZED_MODEL
        input->data.f[i] = g_x_test_test_data[i];
#else
        input->data.uint8[i] = g_x_test_test_data[i];
#endif
    }

    for (int i = 0; i < 10; i++)
    {
        uint64_t start = esp_timer_get_time();
        if (interpreter->Invoke() != kTfLiteOk)
        {
            ESP_LOGE(TAG_M, "Failed to invoke!");
            abort();
        }
        uint64_t end = esp_timer_get_time();

        ESP_LOGI(TAG_M, "Inference time  us: %llu", (end - start));
        total_time += (end - start);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }

    // Calculate average inference time
    uint64_t sum = total_time / 10;
    ESP_LOGI(TAG_M, "Average inference time: %llu us", sum);
}

void get_ram_usage()
{
    long unsigned int arena_used_bytes = interpreter->arena_used_bytes();
    ESP_LOGI(TAG_M, "RAM usage : %lu", arena_used_bytes);
}

extern "C" void app_main(void)
{

    /* Print chip information */
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    printf("This is %s chip with %d CPU core(s), %s%s%s%s\n, ",
           CONFIG_IDF_TARGET,
           chip_info.cores,
           (chip_info.features & CHIP_FEATURE_WIFI_BGN) ? "WiFi/" : "",
           (chip_info.features & CHIP_FEATURE_BT) ? "BT" : "",
           (chip_info.features & CHIP_FEATURE_BLE) ? "BLE" : "",
           (chip_info.features & CHIP_FEATURE_IEEE802154) ? ", 802.15.4 (Zigbee/Thread)" : "");

    init_tflu();
    get_ram_usage();
    run_inference();

    ESP_LOGI(TAG_M, "DONE!");
}
