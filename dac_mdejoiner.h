#ifndef DAC_JOIN_H
#define DAC_JOIN_H

#include <stdint.h>

typedef float data_ty;

// Multi-bit DAC with channel select (8-bit output)
__host__ __device__ void dac_multibit_with_select(data_ty din, data_ty &dout, bool channel_select);

#endif
