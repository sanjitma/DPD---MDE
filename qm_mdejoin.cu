#include <stdint.h>
#include <math.h>

typedef float data_t;
typedef float acc_t;

// This array will be visible to the host (CPU) code.
static const float host_cos_lut[64] = {
    1.0000f, 0.9988f, 0.9951f, 0.9890f, 0.9808f, 0.9703f, 0.9576f, 0.9426f,
    0.9254f, 0.9063f, 0.8854f, 0.8625f, 0.8378f, 0.8115f, 0.7834f, 0.7537f,
    0.7224f, 0.6897f, 0.6557f, 0.6204f, 0.5839f, 0.5464f, 0.5080f, 0.4687f,
    0.4286f, 0.3878f, 0.3464f, 0.3043f, 0.2616f, 0.2185f, 0.1749f, 0.1309f,
    0.0866f, 0.0420f, -0.0027f, -0.0475f, -0.0924f, -0.1374f, -0.1823f, -0.2272f,
    -0.2720f, -0.3167f, -0.3612f, -0.4056f, -0.4497f, -0.4936f, -0.5373f, -0.5807f,
    -0.6237f, -0.6664f, -0.7087f, -0.7505f, -0.7919f, -0.8328f, -0.8732f, -0.9129f,
    -0.9521f, -0.9906f, -1.0284f, -1.0656f, -1.1020f, -1.1376f, -1.1725f, -1.2065f
};

// This __constant__ array will be visible to the device (GPU) code.
__constant__ float device_cos_lut[64] = {
    1.0000f, 0.9988f, 0.9951f, 0.9890f, 0.9808f, 0.9703f, 0.9576f, 0.9426f,
    0.9254f, 0.9063f, 0.8854f, 0.8625f, 0.8378f, 0.8115f, 0.7834f, 0.7537f,
    0.7224f, 0.6897f, 0.6557f, 0.6204f, 0.5839f, 0.5464f, 0.5080f, 0.4687f,
    0.4286f, 0.3878f, 0.3464f, 0.3043f, 0.2616f, 0.2185f, 0.1749f, 0.1309f,
    0.0866f, 0.0420f, -0.0027f, -0.0475f, -0.0924f, -0.1374f, -0.1823f, -0.2272f,
    -0.2720f, -0.3167f, -0.3612f, -0.4056f, -0.4497f, -0.4936f, -0.5373f, -0.5807f,
    -0.6237f, -0.6664f, -0.7087f, -0.7505f, -0.7919f, -0.8328f, -0.8732f, -0.9129f,
    -0.9521f, -0.9906f, -1.0284f, -1.0656f, -1.1020f, -1.1376f, -1.1725f, -1.2065f
};

__host__ __device__ void nco(uint8_t &phase, uint8_t phase_inc, data_t &cos_lo, data_t &sin_lo) {
    uint8_t quadrant = phase >> 6;  // Upper 2 bits for quadrant
    uint8_t index = phase & 0x3F;   // Lower 6 bits for LUT index

    data_t cos_val, sin_val;
    float lut_val, lut_val_sin;

#ifdef __CUDA_ARCH__
    // If compiling for the device, use the __constant__ memory array
    lut_val = device_cos_lut[index];
    lut_val_sin = device_cos_lut[63 - index];
#else
    // If compiling for the host, use the static const host array
    lut_val = host_cos_lut[index];
    lut_val_sin = host_cos_lut[63 - index];
#endif

    switch(quadrant) {
        case 0:
            cos_val = (data_t)lut_val;
            sin_val = (data_t)lut_val_sin;
            break;
        case 1:
            cos_val = (data_t)(-lut_val_sin);
            sin_val = (data_t)lut_val;
            break;
        case 2:
            cos_val = (data_t)(-lut_val);
            sin_val = (data_t)(-lut_val_sin);
            break;
        case 3:
            cos_val = (data_t)lut_val_sin;
            sin_val = (data_t)(-lut_val);
            break;
        default:
            cos_val = 1.0f;
            sin_val = 0.0f;
            break;
    }

    cos_lo = cos_val;
    sin_lo = sin_val;
    phase += phase_inc;
}

// CUDA-compatible QM function
__host__ __device__ data_t digital_qm(data_t I, data_t Q, data_t cos_lo, data_t sin_lo) {
    acc_t mix = (acc_t)I * cos_lo - (acc_t)Q * sin_lo;
    return (data_t)mix;
}
