#include <cuda_runtime.h>
#include <cmath> // For fabsf, sqrtf, etc.
#include "mde.h" // Contains K, MEMORY_DEPTH, data_t, phi_t, acc_t, ccoef_t

// Forward declarations for device functions used within the kernel
__device__ phi_t legendre_poly(int k, phi_t x);
__device__ void compute_phi_all_device(
    const data_t i_in_sample_window[MEMORY_DEPTH],
    const data_t q_in_sample_window[MEMORY_DEPTH],
    phi_t real_phi[K][MEMORY_DEPTH],
    phi_t imag_phi[K][MEMORY_DEPTH]
);

/**
 * @brief Device function to compute Legendre polynomial.
 * This is a helper function for the DPD application.
 *
 * @param k The order of the Legendre polynomial.
 * @param x The input value.
 * @return The value of the k-th Legendre polynomial at x.
 */
__device__ phi_t legendre_poly(int k, phi_t x) {
    if (k == 0) return 1.0f;
    if (k == 1) return x;
    phi_t Pkm2 = 1.0f, Pkm1 = x, Pk = 0.0f;
    for (int n = 2; n <= k; n++) {
        // Recurrence relation for Legendre polynomials
        Pk = ((2.0f * n - 1.0f) * x * Pkm1 - (n - 1.0f) * Pkm2) / n;
        Pkm2 = Pkm1;
        Pkm1 = Pk;
    }
    return Pk;
}

/**
 * @brief Device function to compute basis functions (phi) for a single sample window.
 * This is a helper function for the DPD application.
 *
 * @param i_in_sample_window Array of I input samples for the memory window.
 * @param q_in_sample_window Array of Q input samples for the memory window.
 * @param real_phi Output array for real parts of basis functions.
 * @param imag_phi Output array for imaginary parts of basis functions.
 */
__device__ void compute_phi_all_device(
    const data_t i_in_sample_window[MEMORY_DEPTH],
    const data_t q_in_sample_window[MEMORY_DEPTH],
    phi_t real_phi[K][MEMORY_DEPTH],
    phi_t imag_phi[K][MEMORY_DEPTH]
) {
    for (int tap = 0; tap < MEMORY_DEPTH; ++tap) {
        // Calculate squared magnitude for the current tap input
        phi_t mag_sq = i_in_sample_window[tap] * i_in_sample_window[tap] +
                       q_in_sample_window[tap] * q_in_sample_window[tap];

        for (int k_idx = 0; k_idx < K; ++k_idx) {
            // Compute Legendre polynomial for the current order k and magnitude
            phi_t basis = legendre_poly(k_idx, mag_sq);
            // Store real and imaginary parts of the basis function
            real_phi[k_idx][tap] = i_in_sample_window[tap] * basis;
            imag_phi[k_idx][tap] = q_in_sample_window[tap] * basis;
        }
    }
}

/**
 * @brief CUDA kernel to apply the DPD memory polynomial to a batch of input samples.
 * Each thread processes one output sample.
 *
 * @param i_psf_in_device Input I samples (pulse shaped feedforward).
 * @param q_psf_in_device Input Q samples (pulse shaped feedforward).
 * @param dpd_i_out_device Output I samples after DPD.
 * @param dpd_q_out_device Output Q samples after DPD.
 * @param w_device DPD coefficients (memory polynomial weights).
 * @param num_samples The total number of samples to process.
 */
__global__ void apply_dpd_kernel(
    const data_t *i_psf_in_device,
    const data_t *q_psf_in_device,
    data_t *dpd_i_out_device,
    data_t *dpd_q_out_device,
    const ccoef_t w_device[K][MEMORY_DEPTH], // DPD coefficients
    int num_samples
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; // Global sample index

    if (n < num_samples) {
        // Local buffer to hold the memory window for the current sample
        data_t i_in_sample_window[MEMORY_DEPTH] = {0};
        data_t q_in_sample_window[MEMORY_DEPTH] = {0};

        // Populate the memory window for the current sample 'n'
        // This simulates the memory effect by looking back in time.
        for (int m = 0; m < MEMORY_DEPTH; ++m) {
            int idx_in_history = n - m;
            if (idx_in_history >= 0) {
                i_in_sample_window[m] = i_psf_in_device[idx_in_history];
                q_in_sample_window[m] = q_psf_in_device[idx_in_history];
            } else {
                // Pad with zeros if we are at the beginning of the signal
                i_in_sample_window[m] = 0.0f;
                q_in_sample_window[m] = 0.0f;
            }
        }

        // Compute basis functions for the current memory window
        phi_t real_phi[K][MEMORY_DEPTH];
        phi_t imag_phi[K][MEMORY_DEPTH];
        compute_phi_all_device(i_in_sample_window, q_in_sample_window, real_phi, imag_phi);

        // Apply the DPD memory polynomial using the coefficients
        acc_t z_i_val = 0.0f;
        acc_t z_q_val = 0.0f;

        for (int k_idx = 0; k_idx < K; k_idx++) {
            for (int tap = 0; tap < MEMORY_DEPTH; tap++) {
                // Complex multiplication: (w_real + j*w_imag) * (phi_real + j*phi_imag)
                // Real part: w_real * phi_real - w_imag * phi_imag
                // Imaginary part: w_real * phi_imag + w_imag * phi_real
                z_i_val += w_device[k_idx][tap].real * real_phi[k_idx][tap] -
                           w_device[k_idx][tap].imag * imag_phi[k_idx][tap];
                z_q_val += w_device[k_idx][tap].real * imag_phi[k_idx][tap] +
                           w_device[k_idx][tap].imag * real_phi[k_idx][tap];
            }
        }

        // Store the DPD output for the current sample
        dpd_i_out_device[n] = z_i_val;
        dpd_q_out_device[n] = z_q_val;
    }
}
