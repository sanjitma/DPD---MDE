#ifndef DAC_JOIN_H
#define DAC_JOIN_H

#include <stdint.h>

typedef float data_ty;

// Multi-bit DAC with channel select (8-bit output)
// Wrapped with #ifdef __CUDACC__ to ensure compatibility with both host and device compilers
#ifdef __CUDACC__
__host__ __device__
#endif
void dac_multibit_with_select(data_ty din, data_ty &dout, bool channel_select);

#endif
