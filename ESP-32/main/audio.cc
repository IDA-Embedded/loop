#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "audio.h"

// Static variables
static float gain;
static i2s_chan_handle_t rx_handle;
static int32_t audio_buffer[AUDIO_BUFFER_SAMPLES];
static float audio_frame[AUDIO_BUFFER_SAMPLES] = { 0.0 };
static volatile bool audio_frame_ready = false;

// Forward declarations
static void audio_task(void *args);

/**
 * @brief Initialize audio input and start the audio task. Call this once before calling
 * audio_read().
 * 
 * @param gain   Gain to apply to audio samples. 1.0 is no gain.
*/
void audio_init(float gain)
{
    esp_err_t err;

    // Save gain
    ::gain = gain;

    // Allocate an I2S RX channel
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    err = i2s_new_channel(&chan_cfg, NULL, &rx_handle);
    ESP_ERROR_CHECK(err);

    // Initialize the channel for PDM RX
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = GPIO_NUM_26,
            .ws = GPIO_NUM_32,
            .dout = I2S_GPIO_UNUSED,
            .din = GPIO_NUM_33,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    err = i2s_channel_init_std_mode(rx_handle, &std_cfg);
    ESP_ERROR_CHECK(err);

    // Enable channel
    err = i2s_channel_enable(rx_handle);
    ESP_ERROR_CHECK(err);

    // Start audio task
    xTaskCreate(audio_task, "audio_task", 4096, NULL, 1, NULL);
}

/**
 * @brief Read audio frame. This function blocks until an audio frame is ready.
 * 
 * @return Pointer to audio frame buffer of size AUDIO_BUFFER_SAMPLES.
*/
float* audio_read()
{
    // Wait for audio frame to be ready
    while (!audio_frame_ready)
    {
        vTaskDelay(1);
    }
    audio_frame_ready = false;
    return audio_frame;
}

static void audio_task(void *args)
{
    esp_err_t err;
    size_t bytes_read;
    size_t i;

    while (true)
    {
        // Read audio data into buffer
        err = i2s_channel_read(rx_handle, audio_buffer, sizeof(audio_buffer), &bytes_read, 1000);
        ESP_ERROR_CHECK(err);
        if (bytes_read != sizeof(audio_buffer))
            printf("Read %d bytes instead of %d\n", bytes_read, sizeof(audio_buffer));
        
        // Convert to float audio frame in range [-32767,32767] and apply gain
        for (i = 0; i < AUDIO_BUFFER_SAMPLES; i++)
        {
            float value = (float)(audio_buffer[i] >> 8) * (gain / 256.0);
            if (fabs(value) > 32767.0)
                value = 32767.0 * (value > 0.0 ? 1.0 : -1.0);
            audio_frame[i] = value;
        }

        // Signal audio_read()
        audio_frame_ready = true;
    }
    vTaskDelete(NULL);
}
