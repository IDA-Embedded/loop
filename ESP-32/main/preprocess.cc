#include "preprocess.h"
#include "signal/src/kiss_fft_wrappers/kiss_fft_float.h"

static float feature_buffer[WINDOW_SIZE * SPECTRUM_SIZE];
static float* feature_frame = feature_buffer;
static kiss_fft_float::kiss_fftr_cfg kfft_cfg;
static float max = 0.0;

/**
 * @brief Initialize preprocessing. Call this once before calling preprocess_put_audio() or
 * preprocess_get_features().
*/
bool preprocess_init()
{
    // This code was taken from espressif__esp-tflite-micro/tensorflow/lite/experimental/microfrontend/lib/fft_util.cc.
    // TODO: Maybe use the microfrontent library

    // Ask kissfft how much memory it wants
    size_t scratch_size = 0;
    kfft_cfg = kiss_fft_float::kiss_fftr_alloc(FRAME_SIZE, 0, nullptr, &scratch_size);
    if (kfft_cfg != nullptr) {
        fprintf(stderr, "Kiss memory sizing failed.\n");
        return 0;
    }
    void *scratch = malloc(scratch_size);
    if (scratch == nullptr) {
        fprintf(stderr, "Failed to alloc fft scratch buffer\n");
        return 0;
    }

    // Let kissfft configure the scratch space we just allocated
    kfft_cfg = kiss_fft_float::kiss_fftr_alloc(FRAME_SIZE, 0, scratch, &scratch_size);
    if (kfft_cfg != scratch) {
        fprintf(stderr, "Kiss memory preallocation strategy failed.\n");
        return 0;
    }
    return 1;
}

/**
 * @brief Put an audio frame into the preprocessing pipeline. Call this once per audio frame.
 * Afterwards, call preprocess_get_features() to check if a complete preprocessed feature window
 * is ready for inference.
 * 
 * @param audio_frame Pointer to an array of FRAME_SIZE audio samples.
*/
void preprocess_put_audio(float* audio_frame)
{
    // This function applies the same preprocessing as the following Python code:
    // frame = frame - np.average(frame)
    // frame = frame * np.hamming(FRAME_SIZE)
    // spectral_frame = np.abs(np.fft.rfft(frame))
    // spectral_frame = spectral_frame[0:SPECTRUM_SIZE]
    // spectral_frame = np.log1p(np.abs(spectral_frame))
    // spectral_frame = (spectral_frame - SPECTRUM_MEAN) / SPECTRUM_STD

    // Compute mean and keep track of max
    float mean = 0.0;
    for (int i = 0; i < FRAME_SIZE; i++)
    {
        float value = audio_frame[i];
        mean += value;
        if (fabs(value) > max)
            max = fabs(value);
    }
    mean /= FRAME_SIZE;

    // Subtract mean and apply Hamming window
    for (int i = 0; i < FRAME_SIZE; i++)
    {
        audio_frame[i] = (audio_frame[i] - mean) * (0.54 - 0.46 * cos(2 * M_PI * (float)i / (float)(FRAME_SIZE - 1)));
    }

    // Compute FFT using kiss_fftr
    static kiss_fft_float::kiss_fft_cpx spectrum[FRAME_SIZE / 2 + 1];
    kiss_fft_float::kiss_fftr(kfft_cfg, audio_frame, spectrum);

    // Compute (log(1 + abs(FFT)) - SPECRUM_MEAN) / SPECTRUM_STD
    for (int i = 0; i < SPECTRUM_SIZE; i++)
    {
        float unnormalized = log1p(sqrt(spectrum[i].r * spectrum[i].r + spectrum[i].i * spectrum[i].i));
        float normalized = (unnormalized - SPECTRUM_MEAN) / SPECTRUM_STD;
        *feature_frame++ = normalized;
    }

    // If feature frame is full, reset feature frame pointer
    // This will cause the next call to preprocess_get_features() to provide the features and return true.
    if (feature_frame >= feature_buffer + WINDOW_SIZE * SPECTRUM_SIZE)
    {
        feature_frame = feature_buffer;
    }
}

/**
 * @brief Get features and amplitude. Call this once per audio frame. If it returns true, a full
 * window of features has been copied to the features array.
 * 
 * @param features  Pointer to a array of WINDOW_SIZE * SPECTRUM_SIZE floats.
 * @param amplitude Pointer to a float that will be filled with the amplitude of audio that
 *                  produced the feature window.
 * @return          True if features have been copied and are ready for inference, false otherwise.
 * @todo            This function does not respect stride and will not return overlapping windows.
 *                  Implement a circular buffer to fix this.
*/
bool preprocess_get_features(float* features, float* amplitude)
{
    // If feature frame is not full, return false
    if (feature_frame != feature_buffer)
    {
        return false;
    }

    // Copy feature buffer to features
    memcpy(features, feature_buffer, WINDOW_SIZE * SPECTRUM_SIZE * sizeof(float));

    // Copy amplitude
    *amplitude = max;
    max = 0.0;

    return true;
}
