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
#include "driver/gptimer.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#ifdef CONFIG_MODEL_VERSION_1_0
#pragma message("Using model version 1.0")
#include "model_v1.h"
#include "test_data.h"
#elif defined(CONFIG_MODEL_VERSION_2_0)
#pragma message("Using model version 2.0")
#include "model_v2.h"
#include "test_data2.h"
#endif

// DEFINES
#define TENSOR_ARENA_SIZE (30 * 1024)

// STATIC VARIABLES
/* Log TAG*/
static const char *TAG_M = "Metric Model";
static tflite::MicroInterpreter *interpreter = nullptr;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static gptimer_handle_t gptimer = NULL;

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
    uint64_t time_pr_inference = 0;
    // Get input and output tensors
    TfLiteTensor *input = interpreter->input(0);

    // // Determine input size
    // size_t total_elements = 1;
    // for (int i = 0; i < input->dims->size; ++i)
    // {
    //     total_elements *= input->dims->data[i];
    // }
    // Fill input tensor with data
    // && i < total_elements;
    for (int i = 0; i < g_x_test_test_data_size; i++)
    {
        input->data.f[i] = g_x_test_test_data[i];
    }

    for (int i = 0; i < 10; i++)
    {
        gptimer_start(gptimer);
        // Run inference
        if (interpreter->Invoke() != kTfLiteOk)
        {
            ESP_LOGE(TAG_M, "Failed to invoke!");
            abort();
        }
        gptimer_stop(gptimer);

        gptimer_get_raw_count(gptimer, &time_pr_inference);
        gptimer_set_raw_count(gptimer, 0);
        ESP_LOGI(TAG_M, "Inference time  %llu", time_pr_inference);
        total_time += time_pr_inference;
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }

    // Calculate average inference time
    uint64_t sum = total_time / 10;
    ESP_LOGI(TAG_M, "Average inference time: %llu ms", sum / 1000);
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

    // Init General Purpose Timer
    gptimer_config_t timer_config = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000, // 1MHz, 1 tick=1us
        .intr_priority = 1,
        .flags = {
            .intr_shared = 1,
        }};

    ESP_ERROR_CHECK(gptimer_new_timer(&timer_config, &gptimer));

    // Enable timer
    ESP_ERROR_CHECK(gptimer_enable(gptimer));

    init_tflu();
    get_ram_usage();
    run_inference();

    ESP_LOGI(TAG_M, "DONE!");
}
