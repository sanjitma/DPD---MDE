#ifndef QM_JOIN_H
#define QM_JOIN_H

#include <stdint.h>

typedef float data_t;

// Wrapped with #ifdef __CUDACC__ to ensure compatibility with both host and device compilers
#ifdef __CUDACC__
__host__ __device__
#endif
void nco(uint8_t &phase, uint8_t phase_inc, data_t &cos_lo, data_t &sin_lo);

#ifdef __CUDACC__
__host__ __device__
#endif
data_t digital_qm(data_t I, data_t Q, data_t cos_lo, data_t sin_lo);

#endif
