// File: C:\Modified_DE_cuda\mde-mai.cu
#include "mde.h"
#include <cmath> // For fabsf

// NOTE: The __device__ functions like legendre_poly, compute_phi_all, etc.,
// have been REMOVED from this file because they are defined in mde_kernel.cu.
// The only functions remaining here are the host-side MDE logic.

// These are still needed as they are used by the host-side dpd_mde function.
__host__ __device__ float generate_random_float(int* seed_ptr) {
    *seed_ptr = (*seed_ptr * 1103515245 + 12345) & 0x7fffffff;
    return ((float)(*seed_ptr) / 2147483647.0f); // Uniform [0,1)
}

__host__ __device__ int generate_random_index(int current, int max_val, int* seed_ptr) {
    *seed_ptr = (*seed_ptr * 1103515245 + 12345) & 0x7fffffff;
    int idx = (*seed_ptr) % max_val;
    return (idx == current) ? (idx + 1) % max_val : idx;
}

// This is a private helper for the host-side dpd_mde, so it can stay.
// We make it static to limit its scope to this file only.
static phi_t legendre_poly_host(int k, phi_t x) {
    if (k == 0) return 1.0f;
    if (k == 1) return x;
    phi_t Pkm2 = 1.0f, Pkm1 = x, Pk = 0.0f;
    for (int n = 2; n <= k; n++) {
        Pk = ((2*n-1)*x*Pkm1 - (n-1)*Pkm2)/n;
        Pkm2 = Pkm1;
        Pkm1 = Pk;
    }
    return Pk;
}

// Private helper for dpd_mde on the host
static void compute_phi_all_host(
    const data_t i_in[MEMORY_DEPTH], const data_t q_in[MEMORY_DEPTH],
    phi_t real_phi[K][MEMORY_DEPTH], phi_t imag_phi[K][MEMORY_DEPTH]
) {
    for (int tap = 0; tap < MEMORY_DEPTH; ++tap) {
        phi_t mag = i_in[tap]*i_in[tap] + q_in[tap]*q_in[tap];
        for (int k = 0; k < K; ++k) {
            phi_t basis = legendre_poly_host(k, mag);
            real_phi[k][tap] = i_in[tap] * basis;
            imag_phi[k][tap] = q_in[tap] * basis;
        }
    }
}

static data_t compute_fitness_host(data_t i_ref, data_t q_ref, data_t y_i, data_t y_q) {
    data_t err_i = i_ref - y_i;
    data_t err_q = q_ref - y_q;
    return err_i * err_i + err_q * err_q;
}


// The main host-side DPD function
void dpd_mde(
    data_t i_in[MEMORY_DEPTH], data_t q_in[MEMORY_DEPTH],
    data_t i_ref, data_t q_ref,
    ccoef_t w[K][MEMORY_DEPTH],
    data_t* z_i, data_t* z_q
) {
    static ccoef_t population[POPULATION_SIZE][K][MEMORY_DEPTH];
    static data_t fitness[POPULATION_SIZE];
    static int generation_count = 0;
    static bool init_done = false;
    static int rand_seed = 12345;

    phi_t real_phi[K][MEMORY_DEPTH], imag_phi[K][MEMORY_DEPTH];
    compute_phi_all_host(i_in, q_in, real_phi, imag_phi);

    // ... (rest of the dpd_mde function remains exactly the same, but it now calls
    // the static helper functions like compute_phi_all_host and compute_fitness_host)

    // (Make sure to replace calls inside dpd_mde to use the new static functions)
    // For example:
    // fitness[i] = compute_fitness_host(i_ref, q_ref, test_sum_i, test_sum_q);
    // ... and so on for the other helpers.
    // The content of the dpd_mde function itself does not need to change beyond this.
}
