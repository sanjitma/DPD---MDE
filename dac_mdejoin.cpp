
#include <stdint.h>
#include <cmath>

typedef float data_ty;

// Multi-bit DAC with channel select (8-bit output)
__host__ __device__ void dac_multibit_with_select(data_ty din, data_ty &dout, bool channel_select) {
    // Scale input from [-1, 1) to [-128, 127]
    float quantized = din * 128.0f;
    if (quantized > 127.0f) quantized = 127.0f;
    if (quantized < -128.0f) quantized = -128.0f;
    dout = quantized;
}
