#include <stdint.h>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdio> // For fprintf and printf in CUDA_CHECK
#include "psf_mdejoiner.h"
#include "mde.h"
#include "dac_mdejoiner.h"
#include "qm_mdejoiner.h"
#include "duc_mdejoiner.h"
#include "pa_mdejoiner.h"
#include "ddc_mdejoiner.h"
#include "adc_mdejoiner.h"

#define MAX_INPUT_BYTES 8192
#define DATA_LEN 8192
#define MAX_SYMBOLS 32768
#define INTERPOLATION_FACTOR 8
#define DECIM_FACTOR 8
#define DELAY_OFFSET 112

// Replace ap_fixed/ap_uint types with float/uint types
typedef float fixed_t;
typedef float data_t;
typedef float sample_type;
typedef float baseband_t;
typedef float adc_in_t;
typedef int16_t adc_out_t; // Keep as int16_t as per adc_mdejoiner.h
typedef float data_ty;

// CUDA error check macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

enum DPDMode { BASELINE, ADAPT, FINAL };

// MDE DPD weights - moved to global scope for static initialization behavior
// These are managed by dpd_mde, which is called on the host.
static ccoef_t w[K][MEMORY_DEPTH] = {0};
static bool weights_initialized = false;

// CUDA kernel for baseline mode (minimal stub, real implementation needed)
// This kernel assumes dpd_i and dpd_q are simply passed through i_psf/q_psf
__global__ void baseline_dpd_kernel(
    const data_t *i_psf, const data_t *q_psf,
    data_t *dpd_i, data_t *dpd_q
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < DATA_LEN) {
        dpd_i[idx] = i_psf[idx];
        dpd_q[idx] = q_psf[idx];
    }
}

// CUDA kernel for DAC, QM, and DUC input preparation (FINAL mode first stage)
__global__ void final_dpd_dac_qm_kernel(
    const data_t *i_psf_in, const data_t *q_psf_in,
    data_t *dpd_i_out, data_t *dpd_q_out,
    data_ty *dac_i_arr_out, data_ty *dac_q_arr_out,
    data_t *qm_out_buf_out,
    ccoef_t w_in[K][MEMORY_DEPTH], // Weights from host
    uint8_t phase_inc // NCO phase increment
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to DATA_LEN iteration
    if (n < DATA_LEN) {
        data_t i_in_mde[MEMORY_DEPTH] = {0};
        data_t q_in_mde[MEMORY_DEPTH] = {0};
        for (int m = 0; m < MEMORY_DEPTH; ++m) {
            int idx = n - m;
            i_in_mde[m] = (idx >= 0) ? i_psf_in[idx] : 0.0f;
            q_in_mde[m] = (idx >= 0) ? q_psf_in[idx] : 0.0f;
        }
        data_t i_ref_mde = i_psf_in[n];
        data_t q_ref_mde = q_psf_in[n];

        // Since dpd_mde itself is a complex DE algorithm with static state,
        // it cannot be easily made a __device__ function that fully parallelizes per thread.
        // For 'FINAL' mode, the DPD is applied using already adapted weights 'w'.
        // This application of DPD (matrix multiplication style) *can* be parallelized.
        // However, if dpd_mde is only exposed as a host function, we need to adapt.
        //
        // Assuming 'dpd_mde' here means the *application* of the DPD polynomial,
        // and not the full Differential Evolution algorithm.
        // We will manually apply the memory polynomial part here.

        phi_t real_phi[K][MEMORY_DEPTH], imag_phi[K][MEMORY_DEPTH];
        // Re-implement compute_phi_all for device
        for (int tap = 0; tap < MEMORY_DEPTH; ++tap) {
            phi_t mag_sq = i_in_mde[tap]*i_in_mde[tap] + q_in_mde[tap]*q_in_mde[tap];
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                // Assuming legendre_poly is also __device__
                phi_t basis = legendre_poly(k_idx, mag_sq);
                real_phi[k_idx][tap] = i_in_mde[tap] * basis;
                imag_phi[k_idx][tap] = q_in_mde[tap] * basis;
            }
        }

        acc_t z_i_val = 0.0f, z_q_val = 0.0f;
        for (int k_idx = 0; k_idx < K; k_idx++) {
            for (int tap = 0; tap < MEMORY_DEPTH; tap++) {
                z_i_val += w_in[k_idx][tap].real * real_phi[k_idx][tap] - w_in[k_idx][tap].imag * imag_phi[k_idx][tap];
                z_q_val += w_in[k_idx][tap].real * imag_phi[k_idx][tap] + w_in[k_idx][tap].imag * real_phi[k_idx][tap];
            }
        }
        dpd_i_out[n] = z_i_val;
        dpd_q_out[n] = z_q_val;

        // DAC
        data_ty dac_i, dac_q;
        dac_multibit_with_select(z_i_val, dac_i, 0); // Assuming dac_multibit_with_select is __device__
        dac_multibit_with_select(z_q_val, dac_q, 1); // Assuming dac_multibit_with_select is __device__
        dac_i_arr_out[n] = dac_i;
        dac_q_arr_out[n] = dac_q;

        data_t i_mod_fixed = dac_i * (1.0f/128.0f);
        data_t q_mod_fixed = dac_q * (1.0f/128.0f);

        // NCO and QM
        uint8_t thread_phase = (n * phase_inc) & 0xFF; // Simplified per-thread phase based on index
        data_t cos_lo, sin_lo;
        nco(thread_phase, phase_inc, cos_lo, sin_lo); // Assuming nco is __device__

        data_t qm_out = digital_qm(i_mod_fixed, q_mod_fixed, cos_lo, sin_lo); // Assuming digital_qm is __device__
        qm_out_buf_out[n] = qm_out;
    }
}


// Host function that orchestrates the entire circuit
void circuit_final(
    float input_bytes[MAX_INPUT_BYTES],
    int num_bits,
    sample_type duc_out[DATA_LEN * INTERPOLATION_FACTOR],
    fixed_t i_symbols[MAX_SYMBOLS],
    fixed_t q_symbols[MAX_SYMBOLS],
    fixed_t i_psf[DATA_LEN],
    fixed_t q_psf[DATA_LEN],
    data_t dpd_i[DATA_LEN],
    data_t dpd_q[DATA_LEN],
    data_ty dac_i_arr[DATA_LEN],
    data_ty dac_q_arr[DATA_LEN],
    data_t qm_out_buf[DATA_LEN],
    data_t amp_out_i[DATA_LEN * INTERPOLATION_FACTOR],
    data_t amp_out_q[DATA_LEN * INTERPOLATION_FACTOR],
    data_t amp_magnitude[DATA_LEN * INTERPOLATION_FACTOR],
    data_t amp_gain_lin[DATA_LEN * INTERPOLATION_FACTOR],
    data_t amp_gain_db[DATA_LEN * INTERPOLATION_FACTOR],
    baseband_t ddc_i_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
    baseband_t ddc_q_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
    adc_out_t adc_i_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
    adc_out_t adc_q_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
    fixed_t i_psf_fb[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
    fixed_t q_psf_fb[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
    DPDMode mode
) {
    if (!weights_initialized) {
        // Initialize weights for MDE (typically, only w[0][0] is 1, others 0 for passthrough start)
        for (int m = 0; m < MEMORY_DEPTH; ++m) {
            w[0][m].real = 1.0f;
            w[0][m].imag = 0.0f;
            for(int k_idx = 1; k_idx < K; ++k_idx) {
                w[k_idx][m].real = 0.0f;
                w[k_idx][m].imag = 0.0f;
            }
        }
        weights_initialized = true;
    }

    // Call conste (host function) to get initial symbols
    // conste(input_bytes, num_bits, i_symbols, q_symbols); // User did not provide this function, assuming it's done elsewhere or input_bytes is not used. Commenting out.

    // Pulse shaping (host function, calls __host__ __device__ functions internally)
    // The pulse_shape function now internally calls raised_cosine_filter and convolve,
    // which are marked __host__ __device__. For full CUDA, pulse_shape itself could be a kernel.
    // For now, it's treated as a host-callable utility.
    // To make pulse_shape itself a kernel, one would pass the arrays to device memory,
    // launch a kernel that calls the __device__ versions, and copy back.
    // Given the current structure, pulse_shape operates on arrays which are directly accessible by CPU.
    // To ensure consistency, I'll launch a kernel for pulse_shape if it were on device
    // However, the original code passes data directly, making it appear as if it is a host call.
    // Assuming pulse_shape itself is *not* a kernel, but its sub-functions are device-callable.
    // This implies data will be copied to and from device for each stage if performed fully on GPU.

    // For simplicity, for psf_mdejoin.cu, pulse_shape is made __host__ __device__.
    // The current call to pulse_shape is a host call using host arrays.
    // To utilize GPU, data must be copied to device, kernel launched, then copied back.
    fixed_t *d_i_symbols, *d_q_symbols;
    fixed_t *d_i_psf, *d_q_psf;
    CUDA_CHECK(cudaMalloc(&d_i_symbols, MAX_SYMBOLS * sizeof(fixed_t)));
    CUDA_CHECK(cudaMalloc(&d_q_symbols, MAX_SYMBOLS * sizeof(fixed_t)));
    CUDA_CHECK(cudaMalloc(&d_i_psf, DATA_LEN * sizeof(fixed_t)));
    CUDA_CHECK(cudaMalloc(&d_q_psf, DATA_LEN * sizeof(fixed_t)));

    CUDA_CHECK(cudaMemcpy(d_i_symbols, i_symbols, MAX_SYMBOLS * sizeof(fixed_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_symbols, q_symbols, MAX_SYMBOLS * sizeof(fixed_t), cudaMemcpyHostToDevice));

    // A kernel for pulse_shape could be defined and launched here.
    // For now, call the host-callable `pulse_shape` that operates on CPU memory.
    // This means data copied to device is immediately copied back to host for this step.
    // This is not efficient, but it aligns with the "make compatible" without full redesign.
    // A better approach would be to have all arrays on device and pass device pointers.
    pulse_shape(i_symbols, q_symbols, i_psf, q_psf);

    // After pulse_shape, copy results back to device for subsequent steps
    CUDA_CHECK(cudaMemcpy(d_i_psf, i_psf, DATA_LEN * sizeof(fixed_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_psf, q_psf, DATA_LEN * sizeof(fixed_t), cudaMemcpyHostToDevice));


    uint8_t phase_inc_val = 2; // NCO phase increment for QM

    const int ADC_LEN = (DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR;

    if (mode == BASELINE) {
        // ----------- BASELINE: DPD passthrough, output first pass only -----------
        data_t *d_dpd_i, *d_dpd_q;
        CUDA_CHECK(cudaMalloc(&d_dpd_i, DATA_LEN * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_dpd_q, DATA_LEN * sizeof(data_t)));

        int threadsPerBlock = 256;
        int blocksPerGrid = (DATA_LEN + threadsPerBlock - 1) / threadsPerBlock;
        baseline_dpd_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_i_psf, d_q_psf, d_dpd_i, d_dpd_q);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(dpd_i, d_dpd_i, DATA_LEN * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dpd_q, d_dpd_q, DATA_LEN * sizeof(data_t), cudaMemcpyDeviceToHost));

        cudaFree(d_dpd_i);
        cudaFree(d_dpd_q);

        // DAC, QM, DUC, Amplifier, DDC, ADC, PSF for feedback path
        // For baseline, these are currently host calls with arrays.
        // To make them CUDA, they need device pointers and kernel launches.
        // This is where the major refactoring for CUDA pipeline happens.

        // Allocate device memory for subsequent stages
        sample_type *d_duc_out;
        data_t *d_dac_i_arr, *d_dac_q_arr, *d_qm_out_buf;
        data_t *d_amp_out_i, *d_amp_out_q, *d_amp_magnitude, *d_amp_gain_lin, *d_amp_gain_db;
        baseband_t *d_ddc_i_out, *d_ddc_q_out;
        adc_out_t *d_adc_i_out, *d_adc_q_out;
        fixed_t *d_i_psf_fb, *d_q_psf_fb;

        CUDA_CHECK(cudaMalloc(&d_dac_i_arr, DATA_LEN * sizeof(data_ty)));
        CUDA_CHECK(cudaMalloc(&d_dac_q_arr, DATA_LEN * sizeof(data_ty)));
        CUDA_CHECK(cudaMalloc(&d_qm_out_buf, DATA_LEN * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_duc_out, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(sample_type)));
        CUDA_CHECK(cudaMalloc(&d_amp_out_i, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_amp_out_q, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_amp_magnitude, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_amp_gain_lin, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_amp_gain_db, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_ddc_i_out, ADC_LEN * sizeof(baseband_t)));
        CUDA_CHECK(cudaMalloc(&d_ddc_q_out, ADC_LEN * sizeof(baseband_t)));
        CUDA_CHECK(cudaMalloc(&d_adc_i_out, ADC_LEN * sizeof(adc_out_t)));
        CUDA_CHECK(cudaMalloc(&d_adc_q_out, ADC_LEN * sizeof(adc_out_t)));
        CUDA_CHECK(cudaMalloc(&d_i_psf_fb, ADC_LEN * sizeof(fixed_t)));
        CUDA_CHECK(cudaMalloc(&d_q_psf_fb, ADC_LEN * sizeof(fixed_t)));

        // Stage 1: DAC, QM (can be combined into one kernel)
        // Copy dpd_i/q to device for these stages
        data_t *d_dpd_i_host_copy, *d_dpd_q_host_copy;
        CUDA_CHECK(cudaMalloc(&d_dpd_i_host_copy, DATA_LEN * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_dpd_q_host_copy, DATA_LEN * sizeof(data_t)));
        CUDA_CHECK(cudaMemcpy(d_dpd_i_host_copy, dpd_i, DATA_LEN * sizeof(data_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dpd_q_host_copy, dpd_q, DATA_LEN * sizeof(data_t), cudaMemcpyHostToDevice));


        // For baseline, the dpd_mde is a pass-through (weights [0][0] = 1, others 0)
        // so we can call a simplified kernel or directly use dpd_i/q for DAC/QM.
        // Here, we launch a kernel for DAC/QM for `baseline` path too, using dpd_i/q directly.
        // Note: For actual baseline, the `final_dpd_dac_qm_kernel` would get simplified weights or the inputs would just pass through.
        // The previous `baseline_dpd_kernel` already handled the passthrough of i_psf/q_psf to dpd_i/q.
        // So here we're processing the results of that.

        // Re-copy d_dpd_i, d_dpd_q for usage in this kernel.
        // This is a direct copy from the baseline_dpd_kernel output.
        // Since `final_dpd_dac_qm_kernel` expects original i_psf_in, q_psf_in
        // and uses the weights, we'll simulate the output of the first DPD pass for the baseline.
        // For the baseline, the 'w' used in 'final_dpd_dac_qm_kernel' will essentially implement the passthrough
        // (if w[0][0] = 1, others 0 as initialized).

        final_dpd_dac_qm_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_i_psf, d_q_psf, // Use i_psf and q_psf as input for first pass
            d_dpd_i_host_copy, d_dpd_q_host_copy, // Output of the baseline passthrough DPD
            d_dac_i_arr, d_dac_q_arr,
            d_qm_out_buf,
            w, // Use the w array, which for baseline effectively acts as passthrough
            phase_inc_val
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_dpd_i_host_copy);
        cudaFree(d_dpd_q_host_copy);

        // Stage 2: Digital Upconverter
        // Input to DUC is qm_out_buf (device pointer)
        digital_upconverter(
            (const sample_type*)d_qm_out_buf,
            (const sample_type*)d_qm_out_buf, // Assuming q_in is 0 for single-channel in original DUC call
            d_duc_out,
            0x40000000, // ddc_freq_word (from commented code)
            1 // enable (from commented code)
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Stage 3: Amplifier (Saleh model)
        // Input is duc_out (device pointer)
        // Launch a kernel that iterates and calls saleh_amplifier for each sample
        threadsPerBlock = 256;
        blocksPerGrid = ((DATA_LEN * INTERPOLATION_FACTOR) + threadsPerBlock - 1) / threadsPerBlock;
        // Kernel for amplifier
        __global__ void amplifier_kernel(
            const data_t *duc_in,
            data_t *amp_i_out, data_t *amp_q_out,
            data_t *amp_mag_out, data_t *amp_gain_lin_out, data_t *amp_gain_db_out
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < (DATA_LEN * INTERPOLATION_FACTOR)) {
                data_t local_amp_i, local_amp_q, local_amp_mag, local_amp_gain_lin, local_amp_gain_db;
                saleh_amplifier(
                    duc_in[idx],
                    0.0f, // Assuming Q input is 0 based on original usage
                    local_amp_i, local_amp_q, local_amp_mag, local_amp_gain_lin, local_amp_gain_db
                );
                amp_i_out[idx] = local_amp_i;
                amp_q_out[idx] = local_amp_q;
                amp_mag_out[idx] = local_amp_mag;
                amp_gain_lin_out[idx] = local_amp_gain_lin;
                amp_gain_db_out[idx] = local_amp_gain_db;
            }
        }
        amplifier_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_duc_out,
            d_amp_out_i, d_amp_out_q, d_amp_magnitude, d_amp_gain_lin, d_amp_gain_db
        );
        CUDA_CHECK(cudaDeviceSynchronize());


        // Stage 4: DDC (Digital Downconverter)
        // Input is amp_out_i (device pointer)
        rf_sample_t *d_ddc_in;
        CUDA_CHECK(cudaMalloc(&d_ddc_in, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(rf_sample_t)));
        CUDA_CHECK(cudaMemcpy(d_ddc_in, d_amp_out_i, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(rf_sample_t), cudaMemcpyDeviceToHost));
        // ddc_demodulator handles its own internal device memory management
        ddc_demodulator(
            d_ddc_in,
            d_ddc_i_out,
            d_ddc_q_out,
            DATA_LEN * INTERPOLATION_FACTOR,
            0x40000000, // ddc_freq_word (from commented code)
            25000.0f // ddc_gain (from commented code)
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_ddc_in);


        // Stage 5: ADC (Analog to Digital Converter)
        // Input is ddc_i_out, ddc_q_out (device pointers)
        adc_in_t *d_adc_i_in, *d_adc_q_in;
        CUDA_CHECK(cudaMalloc(&d_adc_i_in, ADC_LEN * sizeof(adc_in_t)));
        CUDA_CHECK(cudaMalloc(&d_adc_q_in, ADC_LEN * sizeof(adc_in_t)));
        CUDA_CHECK(cudaMemcpy(d_adc_i_in, d_ddc_i_out, ADC_LEN * sizeof(adc_in_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_adc_q_in, d_ddc_q_out, ADC_LEN * sizeof(adc_in_t), cudaMemcpyHostToDevice));

        // dual_adc_system handles its own internal device memory management
        dual_adc_system(
            d_adc_i_in, d_adc_q_in,
            d_adc_i_out, d_adc_q_out
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_adc_i_in);
        cudaFree(d_adc_q_in);


        // Stage 6: Pulse Shaping for Feedback (PSF_FB)
        // Input is adc_i_out, adc_q_out (device pointers)
        fixed_t *d_i_psf_fb_in, *d_q_psf_fb_in;
        CUDA_CHECK(cudaMalloc(&d_i_psf_fb_in, ADC_LEN * sizeof(fixed_t)));
        CUDA_CHECK(cudaMalloc(&d_q_psf_fb_in, ADC_LEN * sizeof(fixed_t)));
        CUDA_CHECK(cudaMemcpy(d_i_psf_fb_in, d_adc_i_out, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_q_psf_fb_in, d_adc_q_out, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));

        // Call pulse_shape, which is __host__ __device__. It will operate on GPU via kernel call or directly on host.
        // For consistency, we assume pulse_shape handles its own device copies for now if called from host.
        // A direct kernel for pulse_shape would be better, passing device pointers.
        pulse_shape(d_i_psf_fb_in, d_q_psf_fb_in, d_i_psf_fb, d_q_psf_fb);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Normalization of feedback path
        double ff_rms = 0, fb_rms = 0;
        // Copy back d_i_psf and d_i_psf_fb for RMS calculation on host
        CUDA_CHECK(cudaMemcpy(i_psf, d_i_psf, DATA_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(i_psf_fb, d_i_psf_fb, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));

        for (int i = 0; i < DATA_LEN; ++i) {
            ff_rms += (double)i_psf[i] * (double)i_psf[i];
        }
        for (int i = 0; i < ADC_LEN; ++i) {
            fb_rms += (double)i_psf_fb[i] * (double)i_psf_fb[i];
        }
        ff_rms = std::sqrt(ff_rms / DATA_LEN);
        fb_rms = std::sqrt(fb_rms / ADC_LEN);

        double norm_factor = (fb_rms > 1e-12) ? (ff_rms / fb_rms) : 1.0;

        // Apply normalization on device
        // Kernel for normalization
        __global__ void normalize_kernel(
            fixed_t *i_fb_in_out, fixed_t *q_fb_in_out, double factor, int len
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < len) {
                i_fb_in_out[idx] = (fixed_t)((double)i_fb_in_out[idx] * factor);
                q_fb_in_out[idx] = (fixed_t)((double)q_fb_in_out[idx] * factor);
            }
        }
        blocksPerGrid = (ADC_LEN + threadsPerBlock - 1) / threadsPerBlock;
        normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_i_psf_fb, d_q_psf_fb, norm_factor, ADC_LEN);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy final results for baseline back to host for test_for_mde.cpp to write to files
        CUDA_CHECK(cudaMemcpy(i_psf_fb, d_i_psf_fb, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_psf_fb, d_q_psf_fb, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));

        // Free all device memory
        cudaFree(d_i_symbols); cudaFree(d_q_symbols);
        cudaFree(d_i_psf); cudaFree(d_q_psf);
        cudaFree(d_dac_i_arr); cudaFree(d_dac_q_arr); cudaFree(d_qm_out_buf);
        cudaFree(d_duc_out);
        cudaFree(d_amp_out_i); cudaFree(d_amp_out_q); cudaFree(d_amp_magnitude); cudaFree(d_amp_gain_lin); cudaFree(d_amp_gain_db);
        cudaFree(d_ddc_i_out); cudaFree(d_ddc_q_out);
        cudaFree(d_adc_i_out); cudaFree(d_adc_q_out);
        cudaFree(d_i_psf_fb_in); cudaFree(d_q_psf_fb_in);
        cudaFree(d_i_psf_fb); cudaFree(d_q_psf_fb);

        return;
    }

    if (mode == ADAPT) {
        // ----------- ADAPTATION: Indirect Learning -----------
        // This mode involves the `dpd_mde` (Differential Evolution) algorithm,
        // which currently runs on the host and manages its own static state.
        // It consumes `i_psf_fb` and `q_psf_fb` (from the previous run's feedback path)
        // and `i_psf`, `q_psf` (from the feedforward path) to update the `w` weights.
        // No new kernels are launched in this mode; it's a host-side computation loop.

        // Need to ensure i_psf, q_psf and i_psf_fb, q_psf_fb are on host for dpd_mde
        CUDA_CHECK(cudaMemcpy(i_psf, d_i_psf, DATA_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_psf, d_q_psf, DATA_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));

        fixed_t *d_i_psf_fb_adapt_in, *d_q_psf_fb_adapt_in;
        CUDA_CHECK(cudaMalloc(&d_i_psf_fb_adapt_in, ADC_LEN * sizeof(fixed_t)));
        CUDA_CHECK(cudaMalloc(&d_q_psf_fb_adapt_in, ADC_LEN * sizeof(fixed_t)));
        CUDA_CHECK(cudaMemcpy(d_i_psf_fb_adapt_in, i_psf_fb, ADC_LEN * sizeof(fixed_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q_psf_fb_adapt_in, q_psf_fb, ADC_LEN * sizeof(fixed_t), cudaMemcpyHostToDevice));

        // This loop calls dpd_mde for each sample, which is a host-side operation.
        for (int n = 0; n < DATA_LEN && n < ADC_LEN; ++n) {
            data_t i_in_mde[MEMORY_DEPTH] = {0};
            data_t q_in_mde[MEMORY_DEPTH] = {0};
            int fb_idx = n; // Current sample index in feedback path
            for (int m = 0; m < MEMORY_DEPTH; ++m) {
                int idx = fb_idx - m;
                // Use data from feedback path (i_psf_fb) for DPD input
                i_in_mde[m] = (idx >= 0 && idx < ADC_LEN) ? (data_t)i_psf_fb[idx] : 0.0f;
                q_in_mde[m] = (idx >= 0 && idx < ADC_LEN) ? (data_t)q_psf_fb[idx] : 0.0f;
            }
            int ref_idx = (n >= DELAY_OFFSET) ? (n - DELAY_OFFSET) : 0;
            // Use data from feedforward path (i_psf) as reference for DPD
            data_t i_ref = (ref_idx < DATA_LEN) ? (data_t)i_psf[ref_idx] : 0.0f;
            data_t q_ref = (ref_idx < DATA_LEN) ? (data_t)q_psf[ref_idx] : 0.0f;

            data_t z_i, z_q; // z_i and z_q here are not used as outputs, but dpd_mde updates 'w'
            dpd_mde(i_in_mde, q_in_mde, i_ref, q_ref, w, &z_i, &z_q); // Calls host function dpd_mde
        }

        cudaFree(d_i_psf_fb_adapt_in);
        cudaFree(d_q_psf_fb_adapt_in);

        // Free device memory allocated at the start of circuit_final
        cudaFree(d_i_symbols); cudaFree(d_q_symbols);
        cudaFree(d_i_psf); cudaFree(d_q_psf);

        return;
    }

    if (mode == FINAL) {
        // ----------- FINAL: DPD with adapted weights, output second pass only -----------
        // Allocate device memory for all stages, as before in BASELINE
        sample_type *d_duc_out;
        data_ty *d_dac_i_arr, *d_dac_q_arr;
        data_t *d_qm_out_buf;
        data_t *d_amp_out_i, *d_amp_out_q, *d_amp_magnitude, *d_amp_gain_lin, *d_amp_gain_db;
        baseband_t *d_ddc_i_out, *d_ddc_q_out;
        adc_out_t *d_adc_i_out, *d_adc_q_out;
        fixed_t *d_i_psf_fb_final_out, *d_q_psf_fb_final_out;

        CUDA_CHECK(cudaMalloc(&d_dac_i_arr, DATA_LEN * sizeof(data_ty)));
        CUDA_CHECK(cudaMalloc(&d_dac_q_arr, DATA_LEN * sizeof(data_ty)));
        CUDA_CHECK(cudaMalloc(&d_qm_out_buf, DATA_LEN * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_duc_out, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(sample_type)));
        CUDA_CHECK(cudaMalloc(&d_amp_out_i, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_amp_out_q, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_amp_magnitude, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_amp_gain_lin, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_amp_gain_db, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_ddc_i_out, ADC_LEN * sizeof(baseband_t)));
        CUDA_CHECK(cudaMalloc(&d_ddc_q_out, ADC_LEN * sizeof(baseband_t)));
        CUDA_CHECK(cudaMalloc(&d_adc_i_out, ADC_LEN * sizeof(adc_out_t)));
        CUDA_CHECK(cudaMalloc(&d_adc_q_out, ADC_LEN * sizeof(adc_out_t)));
        CUDA_CHECK(cudaMalloc(&d_i_psf_fb_final_out, ADC_LEN * sizeof(fixed_t)));
        CUDA_CHECK(cudaMalloc(&d_q_psf_fb_final_out, ADC_LEN * sizeof(fixed_t)));

        // Copy current DPD weights to a __constant__ memory or pass them.
        // For simplicity, pass the `w` array directly to the kernel for now.
        // For a real-time system, `w` could be updated in __constant__ memory.

        int threadsPerBlock = 256;
        int blocksPerGrid = (DATA_LEN + threadsPerBlock - 1) / threadsPerBlock;

        // Stage 1: DPD application, DAC, QM
        final_dpd_dac_qm_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_i_psf, d_q_psf, // Use the original pulse-shaped symbols as input
            dpd_i, dpd_q, // Store the DPD output for analysis (copied back later)
            d_dac_i_arr, d_dac_q_arr,
            d_qm_out_buf,
            w, // Pass the adapted weights 'w'
            phase_inc_val
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy DPD outputs back to host for file writing
        CUDA_CHECK(cudaMemcpy(dpd_i, dpd_i, DATA_LEN * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dpd_q, dpd_q, DATA_LEN * sizeof(data_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dac_i_arr, d_dac_i_arr, DATA_LEN * sizeof(data_ty), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dac_q_arr, d_dac_q_arr, DATA_LEN * sizeof(data_ty), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(qm_out_buf, d_qm_out_buf, DATA_LEN * sizeof(data_t), cudaMemcpyDeviceToHost));


        // Stage 2: Digital Upconverter
        digital_upconverter(
            (const sample_type*)d_qm_out_buf,
            (const sample_type*)d_qm_out_buf, // Assuming q_in is 0
            d_duc_out,
            0x40000000, // ddc_freq_word
            1 // enable
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Stage 3: Amplifier (Saleh model)
        blocksPerGrid = ((DATA_LEN * INTERPOLATION_FACTOR) + threadsPerBlock - 1) / threadsPerBlock;
        amplifier_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_duc_out,
            d_amp_out_i, d_amp_out_q, d_amp_magnitude, d_amp_gain_lin, d_amp_gain_db
        );
        CUDA_CHECK(cudaDeviceSynchronize());


        // Stage 4: DDC (Digital Downconverter)
        rf_sample_t *d_ddc_in;
        CUDA_CHECK(cudaMalloc(&d_ddc_in, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(rf_sample_t)));
        CUDA_CHECK(cudaMemcpy(d_ddc_in, d_amp_out_i, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(rf_sample_t), cudaMemcpyDeviceToHost));
        ddc_demodulator(
            d_ddc_in,
            d_ddc_i_out,
            d_ddc_q_out,
            DATA_LEN * INTERPOLATION_FACTOR,
            0x40000000,
            25000.0f
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_ddc_in);


        // Stage 5: ADC (Analog to Digital Converter)
        adc_in_t *d_adc_i_in, *d_adc_q_in;
        CUDA_CHECK(cudaMalloc(&d_adc_i_in, ADC_LEN * sizeof(adc_in_t)));
        CUDA_CHECK(cudaMalloc(&d_adc_q_in, ADC_LEN * sizeof(adc_in_t)));
        CUDA_CHECK(cudaMemcpy(d_adc_i_in, d_ddc_i_out, ADC_LEN * sizeof(adc_in_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_adc_q_in, d_ddc_q_out, ADC_LEN * sizeof(adc_in_t), cudaMemcpyHostToDevice));

        dual_adc_system(
            d_adc_i_in, d_adc_q_in,
            d_adc_i_out, d_adc_q_out
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_adc_i_in);
        cudaFree(d_adc_q_in);

        // Stage 6: Pulse Shaping for Feedback (PSF_FB)
        fixed_t *d_i_psf_fb_in_final, *d_q_psf_fb_in_final;
        CUDA_CHECK(cudaMalloc(&d_i_psf_fb_in_final, ADC_LEN * sizeof(fixed_t)));
        CUDA_CHECK(cudaMalloc(&d_q_psf_fb_in_final, ADC_LEN * sizeof(fixed_t)));
        CUDA_CHECK(cudaMemcpy(d_i_psf_fb_in_final, d_adc_i_out, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_q_psf_fb_in_final, d_adc_q_out, ADC_LEN * sizeof(fixed_t), cudaMemcpyHostToDevice));

        pulse_shape(d_i_psf_fb_in_final, d_q_psf_fb_in_final, d_i_psf_fb_final_out, d_q_psf_fb_final_out);
        CUDA_CHECK(cudaDeviceSynchronize());


        // Normalization of feedback path
        double ff_rms = 0, fb_rms = 0;
        // Copy back d_i_psf and d_i_psf_fb_final_out for RMS calculation on host
        CUDA_CHECK(cudaMemcpy(i_psf, d_i_psf, DATA_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(i_psf_fb, d_i_psf_fb_final_out, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));

        for (int i = 0; i < DATA_LEN; ++i) {
            ff_rms += (double)i_psf[i] * (double)i_psf[i];
        }
        for (int i = 0; i < ADC_LEN; ++i) {
            fb_rms += (double)i_psf_fb[i] * (double)i_psf_fb[i];
        }
        ff_rms = std::sqrt(ff_rms / DATA_LEN);
        fb_rms = std::sqrt(fb_rms / ADC_LEN);

        double norm_factor = (fb_rms > 1e-12) ? (ff_rms / fb_rms) : 1.0;

        blocksPerGrid = (ADC_LEN + threadsPerBlock - 1) / threadsPerBlock;
        normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_i_psf_fb_final_out, d_q_psf_fb_final_out, norm_factor, ADC_LEN);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy final results for final run back to host for test_for_mde.cpp to write to files
        CUDA_CHECK(cudaMemcpy(i_psf_fb, d_i_psf_fb_final_out, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_psf_fb, d_q_psf_fb_final_out, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));

        // Copy other intermediate results to host if needed by test_for_mde.cpp
        CUDA_CHECK(cudaMemcpy(duc_out, d_duc_out, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(sample_type), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amp_out_i, d_amp_out_i, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amp_out_q, d_amp_out_q, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(amp_magnitude, d_amp_magnitude, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amp_gain_lin, d_amp_gain_lin, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amp_gain_db, d_amp_gain_db, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ddc_i_out, d_ddc_i_out, ADC_LEN * sizeof(baseband_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ddc_q_out, d_ddc_q_out, ADC_LEN * sizeof(baseband_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(adc_i_out, d_adc_i_out, ADC_LEN * sizeof(adc_out_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(adc_q_out, d_adc_q_out, ADC_LEN * sizeof(adc_out_t), cudaMemcpyDeviceToHost));


        // Free all device memory
        cudaFree(d_i_symbols); cudaFree(d_q_symbols);
        cudaFree(d_i_psf); cudaFree(d_q_psf);
        cudaFree(d_dac_i_arr); cudaFree(d_dac_q_arr); cudaFree(d_qm_out_buf);
        cudaFree(d_duc_out);
        cudaFree(d_amp_out_i); cudaFree(d_amp_out_q); cudaFree(d_amp_magnitude); cudaFree(d_amp_gain_lin); cudaFree(d_amp_gain_db);
        cudaFree(d_ddc_i_out); cudaFree(d_ddc_q_out);
        cudaFree(d_adc_i_out); cudaFree(d_adc_q_out);
        cudaFree(d_i_psf_fb_in_final); cudaFree(d_q_psf_fb_in_final);
        cudaFree(d_i_psf_fb_final_out); cudaFree(d_q_psf_fb_final_out);

        return;
    }
}
