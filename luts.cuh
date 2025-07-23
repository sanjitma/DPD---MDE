// File: C:\Modified_DE_cuda\luts.cuh
#ifndef LUTS_CUH
#define LUTS_CUH

#include <cuda_runtime.h>

#define NCO_LUT_SIZE 1024

// Use 'extern' to declare that these constant arrays are defined elsewhere.
// Any file that includes this header will know about the LUTs without redefining them.
extern __constant__ float d_sine_lut[NCO_LUT_SIZE];
extern __constant__ float d_cosine_lut[NCO_LUT_SIZE];

#endif // LUTS_CUH
