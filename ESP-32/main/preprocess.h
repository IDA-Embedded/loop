#pragma once

#include <stdint.h>
#ifdef MODEL_VERSION_1
  #include "model_v1.h"
#else
  #include "model.h"
#endif

bool preprocess_init();
void preprocess_put_audio(float* audio_frame);
bool preprocess_get_features(float* features, float* amplitude);