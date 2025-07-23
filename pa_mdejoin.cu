#include <stdint.h>
#include <cmath>

#define DATA_LEN 8192
#define NUM_WEIGHTS 41
#define SPS 25
#define ALPHA 0.5

typedef float fixed_t;

// CUDA-compatible raised cosine filter
// FIXED: Removed "default" and replaced __declspec with __host__ __device__
__host__ __device__ void raised_cosine_filter(fixed_t rc[NUM_WEIGHTS]) {
    const float PI = 3.14159f;
    const float ALPHA_FIXED = ALPHA;
    const float EPS = 1e-5f;
    const float EPS_X = 1e-3f;
    const float SCALE = 0.9999f;

    int mid = NUM_WEIGHTS / 2;
    // The 'sum' variable was unused for normalization, so it's safe to remove or ignore.
    // float sum = 0.001f;

    for (int i = 0; i < NUM_WEIGHTS; i++) {
        float idx = float(i - mid);
        float x = SCALE * idx / float(SPS);
        float pi_x = PI * x;

        float sinc = (fabsf(x) < EPS_X) ? 1.0f : sinf(pi_x) / pi_x;

        float denom = 1.0f - 4.0f * ALPHA_FIXED * ALPHA_FIXED * x * x;
        if (fabsf(denom) < EPS)
            denom = EPS;

        float angle = PI * ALPHA_FIXED * x;
        float cos_part = cosf(angle);

        rc[i] = sinc * (cos_part / denom);
        // sum += rc[i];
    }

    // Normalize so the peak (max absolute value) is 1
    float max_abs = 0.0f;
    for (int i = 0; i < NUM_WEIGHTS; i++) {
        if (fabsf(rc[i]) > max_abs)
            max_abs = fabsf(rc[i]);
    }
    if (fabsf(max_abs) < EPS)
        max_abs = 1.0f;

    for (int i = 0; i < NUM_WEIGHTS; i++) {
        rc[i] = rc[i] / max_abs;
    }
}

// CUDA-compatible convolve function
__host__ __device__ void convolve(const fixed_t data[DATA_LEN], const fixed_t filter[NUM_WEIGHTS], fixed_t result[DATA_LEN]) {
    int mid = NUM_WEIGHTS / 2;

    for (int i = 0; i < DATA_LEN; i++) {
        fixed_t acc = 0;
        for (int j = 0; j < NUM_WEIGHTS; j++) {
            int k = i - mid + j;
            if (k >= 0 && k < DATA_LEN)
                acc += data[k] * filter[j];
        }
        result[i] = acc;
    }
}

// CUDA-compatible pulse shaping function
__host__ __device__ void pulse_shape(fixed_t i_data[DATA_LEN], fixed_t q_data[DATA_LEN], fixed_t i_out[DATA_LEN], fixed_t q_out[DATA_LEN]) {
    fixed_t rc_filter[NUM_WEIGHTS];
    raised_cosine_filter(rc_filter);
    convolve(i_data, rc_filter, i_out);
    convolve(q_data, rc_filter, q_out);
}
