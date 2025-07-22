#include "mde.h"
#include <cmath> // For fabsf

// Forward declarations for CUDA compatibility - already present and good.
__host__ __device__ phi_t legendre_poly(int k, phi_t x);
__host__ __device__ void compute_phi_all(const data_t i_in[MEMORY_DEPTH], const data_t q_in[MEMORY_DEPTH], phi_t real_phi[K][MEMORY_DEPTH], phi_t imag_phi[K][MEMORY_DEPTH]);
__host__ __device__ float generate_random_float(int* seed_ptr);
__host__ __device__ int generate_random_index(int current, int max_val, int* seed_ptr);
__host__ __device__ data_t compute_fitness(data_t i_ref, data_t q_ref, data_t y_i, data_t y_q);

// Orthogonal polynomial basis (Legendre)
__host__ __device__ phi_t legendre_poly(int k, phi_t x) {
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

// Compute basis functions for all taps
__host__ __device__ void compute_phi_all(
    const data_t i_in[MEMORY_DEPTH], const data_t q_in[MEMORY_DEPTH],
    phi_t real_phi[K][MEMORY_DEPTH], phi_t imag_phi[K][MEMORY_DEPTH]
) {
    for (int tap = 0; tap < MEMORY_DEPTH; ++tap) {
        phi_t mag = i_in[tap]*i_in[tap] + q_in[tap]*q_in[tap];
        for (int k = 0; k < K; ++k) {
            phi_t basis = legendre_poly(k, mag);
            real_phi[k][tap] = i_in[tap] * basis;
            imag_phi[k][tap] = q_in[tap] * basis;
        }
    }
}

// Fitness function (mean square error)
__host__ __device__ data_t compute_fitness(data_t i_ref, data_t q_ref, data_t y_i, data_t y_q) {
    data_t err_i = i_ref - y_i;
    data_t err_q = q_ref - y_q;
    return err_i * err_i + err_q * err_q;
}

// Simple random number generator for CUDA/CPU
__host__ __device__ float generate_random_float(int* seed_ptr) {
    *seed_ptr = (*seed_ptr * 1103515245 + 12345) & 0x7fffffff;
    return ((float)(*seed_ptr) / 2147483647.0f); // Uniform [0,1)
}

__host__ __device__ int generate_random_index(int current, int max_val, int* seed_ptr) {
    *seed_ptr = (*seed_ptr * 1103515245 + 12345) & 0x7fffffff;
    int idx = (*seed_ptr) % max_val;
    return (idx == current) ? (idx + 1) % max_val : idx;
}

// DPD function - intended to be called from host, manages static state
// If this function were to be parallelized on the device, its static state
// (population, fitness) would need to be managed in __device__ memory
// or passed as explicit parameters, and the DE algorithm itself would be
// implemented as a kernel. For this request, it remains a host-side function.
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
    compute_phi_all(i_in, q_in, real_phi, imag_phi);

    // 1. Initialize population
    if (!init_done) {
        for (int i = 0; i < POPULATION_SIZE; i++) {
            for (int k = 0; k < K; k++) {
                for (int tap = 0; tap < MEMORY_DEPTH; tap++) {
                    data_t perturbation_real = (generate_random_float(&rand_seed) - 0.5f) * 2.0f;
                    data_t perturbation_imag = (generate_random_float(&rand_seed) - 0.5f) * 2.0f;
                    if (k == 0 && tap == 0) {
                        population[i][k][tap].real = 1.0f + perturbation_real;
                        population[i][k][tap].imag = perturbation_imag;
                    } else {
                        population[i][k][tap].real = perturbation_real;
                        population[i][k][tap].imag = perturbation_imag;
                    }
                    fitness[i] = 1000.0f; // Initialize with high fitness
                }
            }
        }
        init_done = true;
    }

    // 2. Find best individual
    int best_idx = 0;
    data_t best_fitness = fitness[0];
    for (int i = 1; i < POPULATION_SIZE; i++) {
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            best_idx = i;
        }
    }

    // 3. Compute DPD output z(n) using best individual (this is the actual DPD application)
    // This part operates on the current input sample 'i_in', 'q_in' using the best weights.
    acc_t sum_i = 0.0f, sum_q = 0.0f;
    for (int k = 0; k < K; k++) {
        for (int tap = 0; tap < MEMORY_DEPTH; tap++) {
            sum_i += population[best_idx][k][tap].real * real_phi[k][tap] - population[best_idx][k][tap].imag * imag_phi[k][tap];
            sum_q += population[best_idx][k][tap].real * imag_phi[k][tap] + population[best_idx][k][tap].imag * real_phi[k][tap];
        }
    }
    *z_i = sum_i;
    *z_q = sum_q;

    // 4. Evaluate fitness for all population members
    // This step is part of the DE algorithm, evaluating how well each set of weights
    // in the population would perform for the current input.
    for (int i = 0; i < POPULATION_SIZE; i++) {
        acc_t test_sum_i = 0.0f, test_sum_q = 0.0f;
        for (int k = 0; k < K; k++) {
            for (int tap = 0; tap < MEMORY_DEPTH; tap++) {
                test_sum_i += population[i][k][tap].real * real_phi[k][tap] - population[i][k][tap].imag * imag_phi[k][tap];
                test_sum_q += population[i][k][tap].real * imag_phi[k][tap] + population[i][k][tap].imag * real_phi[k][tap];
            }
        }
        fitness[i] = compute_fitness(i_ref, q_ref, test_sum_i, test_sum_q);
    }

    // 5. Update weights after full DE cycle
    // This is where the best weights found over a set of generations are committed.
    generation_count++;
    if (generation_count >= MAX_GENERATIONS) {
        for (int k = 0; k < K; k++) {
            for (int tap = 0; tap < MEMORY_DEPTH; tap++) {
                w[k][tap] = population[best_idx][k][tap];
            }
        }
        generation_count = 0;
    }

    // 6. DE Mutation, Crossover, Selection (always runs per call to dpd_mde)
    // This is the core of the Differential Evolution algorithm.
    for (int i = 0; i < POPULATION_SIZE; i++) {
        int r1 = generate_random_index(i, POPULATION_SIZE, &rand_seed);
        int r2 = generate_random_index(r1, POPULATION_SIZE, &rand_seed);
        int r3 = generate_random_index(r2, POPULATION_SIZE, &rand_seed);

        ccoef_t trial[K][MEMORY_DEPTH];

        // Mutation and Crossover to create a 'trial' vector
        for (int k = 0; k < K; k++) {
            for (int tap = 0; tap < MEMORY_DEPTH; tap++) {
                trial[k][tap].real = population[r1][k][tap].real + F_SCALE * (population[r2][k][tap].real - population[r3][k][tap].real);
                trial[k][tap].imag = population[r1][k][tap].imag + F_SCALE * (population[r2][k][tap].imag - population[r3][k][tap].imag);

                int cross_rand_val = generate_random_index(0, 1000, &rand_seed); // Using generate_random_index to get a random value [0,999]
                data_t rand_val = (data_t)cross_rand_val * 0.001f; // Scale to [0, 0.999]
                if (rand_val > CR_PROB) {
                    trial[k][tap] = population[i][k][tap]; // Apply crossover based on CR_PROB
                }
            }
        }

        // Evaluate the trial vector's fitness
        acc_t trial_sum_i = 0.0f, trial_sum_q = 0.0f;
        for (int k = 0; k < K; k++) {
            for (int tap = 0; tap < MEMORY_DEPTH; tap++) {
                trial_sum_i += trial[k][tap].real * real_phi[k][tap] - trial[k][tap].imag * imag_phi[k][tap];
                trial_sum_q += trial[k][tap].real * imag_phi[k][tap] + trial[k][tap].imag * real_phi[k][tap];
            }
        }
        data_t trial_fitness = compute_fitness(i_ref, q_ref, trial_sum_i, trial_sum_q);

        // Selection: if trial is better, replace current individual in population
        if (trial_fitness < fitness[i]) {
            for (int k = 0; k < K; k++) {
                for (int tap = 0; tap < MEMORY_DEPTH; tap++) {
                    population[i][k][tap] = trial[k][tap];
                }
            }
            fitness[i] = trial_fitness;
        }
    }
}
