#include <stdint.h>
#include <math.h>

typedef float data_t;      // 16-bit, 4 integer bits, 12 fractional bits
typedef float acc_t; // 32-bit, 4 integer bits, 28 fractional bits

// Move LUT to device constant memory for CUDA, and static const for host
#ifdef __CUDA_ARCH__
__device__ __constant__ float cos_lut[64] = {
    1.0000, 0.9988, 0.9951, 0.9890, 0.9808, 0.9703, 0.9576, 0.9426,
    0.9254, 0.9063, 0.8854, 0.8625, 0.8378, 0.8115, 0.7834, 0.7537,
    0.7224, 0.6897, 0.6557, 0.6204, 0.5839, 0.5464, 0.5080, 0.4687,
    0.4286, 0.3878, 0.3464, 0.3043, 0.2616, 0.2185, 0.1749, 0.1309,
    0.0866, 0.0420, -0.0027, -0.0475, -0.0924, -0.1374, -0.1823, -0.2272,
    -0.2720, -0.3167, -0.3612, -0.4056, -0.4497, -0.4936, -0.5373, -0.5807,
    -0.6237, -0.6664, -0.7087, -0.7505, -0.7919, -0.8328, -0.8732, -0.9129,
    -0.9521, -0.9906, -1.0284, -1.0656, -1.1020, -1.1376, -1.1725, -1.2065
};
#else
static const float host_cos_lut[64] = {
    1.0000, 0.9988, 0.9951, 0.9890, 0.9808, 0.9703, 0.9576, 0.9426,
    0.9254, 0.9063, 0.8854, 0.8625, 0.8378, 0.8115, 0.7834, 0.7537,
    0.7224, 0.6897, 0.6557, 0.6204, 0.5839, 0.5464, 0.5080, 0.4687,
    0.4286, 0.3878, 0.3464, 0.3043, 0.2616, 0.2185, 0.1749, 0.1309,
    0.0866, 0.0420, -0.0027, -0.0475, -0.0924, -0.1374, -0.1823, -0.2272,
    -0.2720, -0.3167, -0.3612, -0.4056, -0.4497, -0.4936, -0.5373, -0.5807,
    -0.6237, -0.6664, -0.7087, -0.7505, -0.7919, -0.8328, -0.8732, -0.9129,
    -0.9521, -0.9906, -1.0284, -1.0656, -1.1020, -1.1376, -1.1725, -1.2065
};
#endif

__host__ __device__ void nco(uint8_t &phase, uint8_t phase_inc, data_t &cos_lo, data_t &sin_lo) {
    uint8_t quadrant = phase >> 6;  // Upper 2 bits for quadrant
    uint8_t index = phase & 0x3F;   // Lower 6 bits for LUT index

    data_t cos_val, sin_val;
    float lut_val;
#ifdef __CUDA_ARCH__
    lut_val = cos_lut[index];
#else
    lut_val = host_cos_lut[index];
#endif

    switch(quadrant) {
        case 0:
            cos_val = (data_t)lut_val;
            sin_val = (data_t)
#ifdef __CUDA_ARCH__
                cos_lut[63-index];
#else
                host_cos_lut[63-index];
#endif
            break;
        case 1:
            cos_val = (data_t)(
#ifdef __CUDA_ARCH__
                -cos_lut[63-index]
#else
                -host_cos_lut[63-index]
#endif
            );
            sin_val = (data_t)lut_val;
            break;
        case 2:
            cos_val = (data_t)(-lut_val);
            sin_val = (data_t)(
#ifdef __CUDA_ARCH__
                -cos_lut[63-index]
#else
                -host_cos_lut[63-index]
#endif
            );
            break;
        case 3:
            cos_val = (data_t)
#ifdef __CUDA_ARCH__
                cos_lut[63-index];
#else
                host_cos_lut[63-index];
#endif
            sin_val = (data_t)(-lut_val);
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
