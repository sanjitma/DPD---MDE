#ifndef AMPLIFIER_HERE_H
#define AMPLIFIER_HERE_H

#include "mde.h"

// Saleh amplifier model
// Added __host__ __device__ to the declaration to match the definition
// Wrapped with #ifdef __CUDACC__ to ensure compatibility with both host and device compilers
#ifdef __CUDACC__
__host__ __device__
#endif
void saleh_amplifier(
    data_t in_i,
    data_t in_q,
    data_t& out_i,
    data_t& out_q,
    data_t& magnitude,
    data_t& gain_lin,
    data_t& gain_db
);

#endif // AMPLIFIER_HERE_H
