// File: C:\Modified_DE_cuda\luts.cu

#include "luts.cuh"

// This is the one and only definition of the LUT arrays.
__constant__ float d_sine_lut[NCO_LUT_SIZE] = {
    #include "sin_lut.h"
};

__constant__ float d_cosine_lut[NCO_LUT_SIZE] = {
    #include "cos_lut.h"
};
