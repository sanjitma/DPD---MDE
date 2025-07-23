// File: C:\Modified_DE_cuda\mde_kernel.cuh
#ifndef MDE_KERNEL_CUH
#define MDE_KERNEL_CUH

#include "mde.h" // For types like data_t, ccoef_t, etc.

// Declaration for the kernel function. Other .cu files will include this
// header to learn about this kernel without seeing its full code.
__global__ void apply_dpd_kernel(
    const data_t *i_psf_in_device,
    const data_t *q_psf_in_device,
    data_t *dpd_i_out_device,
    data_t *dpd_q_out_device,
    const ccoef_t w_device[K][MEMORY_DEPTH],
    int num_samples
);

#endif // MDE_KERNEL_CUH
