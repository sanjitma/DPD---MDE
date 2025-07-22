#include <stdint.h>
#include <cmath> // For float functions like fabsf, if needed.

typedef float data_ty;

// Multi-bit DAC with channel select (8-bit output)
// IMPORTANT: Ensure __host__ __device__ is present here on the definition.
__host__ __device__ void dac_multibit_with_select(data_ty din, data_ty &dout, bool channel_select) {
    // Scale input from [-1, 1) to [-128, 127]
    // Using explicit 'f' suffix for float literals to ensure float precision.
    float quantized = din * 128.0f;

    // Clamp the value to the valid range [-128.0f, 127.0f]
    if (quantized > 127.0f) {
        quantized = 127.0f;
    }
    if (quantized < -128.0f) {
        quantized = -128.0f;
    }
    
    dout = quantized;
}
