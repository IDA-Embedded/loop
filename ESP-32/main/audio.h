#pragma once

#define AUDIO_BUFFER_SAMPLES 256
#define AUDIO_SAMPLE_RATE 16000

void audio_init(float gain);
float* audio_read();