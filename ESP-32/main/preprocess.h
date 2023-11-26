#pragma once

#include <stdint.h>
#include "model.h"

bool preprocess_init();
void preprocess_put_audio(float* audio_frame);
bool preprocess_get_features(float* features, float* amplitude);