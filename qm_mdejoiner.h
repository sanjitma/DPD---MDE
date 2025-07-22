#ifndef QM_JOIN_H
#define QM_JOIN_H

#include <stdint.h>

typedef float data_t;

__host__ __device__ void nco(uint8_t &phase, uint8_t phase_inc, data_t &cos_lo, data_t &sin_lo);
__host__ __device__ data_t digital_qm(data_t I, data_t Q, data_t cos_lo, data_t sin_lo);

#endif
