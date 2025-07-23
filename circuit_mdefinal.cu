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
#include "mde_kernel.cuh" // Include the new kernel file

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
// This kernel now receives the DPD output (dpd_i_in, dpd_q_in) as its direct input
// instead of performing the DPD calculation itself.
__global__ void dac_qm_kernel(
    const data_t *dpd_i_in, const data_t *dpd_q_in,
    data_ty *dac_i_arr_out, data_ty *dac_q_arr_out,
    data_t *qm_out_buf_out,
    uint8_t phase_inc // NCO phase increment
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to DATA_LEN iteration
    if (n < DATA_LEN) {
        // DAC
        data_ty dac_i, dac_q;
        dac_multibit_with_select(dpd_i_in[n], dac_i, 0); // Assuming dac_multibit_with_select is __device__
        dac_multibit_with_select(dpd_q_in[n], dac_q, 1); // Assuming dac_multibit_with_select is __device__
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

// CUDA kernel for Amplifier (Saleh model)
__global__ void amplifier_kernel(
    const data_t *duc_in,
    data_t *amp_i_out, data_t *amp_q_out,
    data_t *amp_mag_out, data_t *amp_gain_lin_out, data_t *amp_gain_db_out,
    int total_samples
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_samples) {
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

// CUDA kernel for normalization
__global__ void normalize_kernel(
    fixed_t *i_fb_in_out, fixed_t *q_fb_in_out, double factor, int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        i_fb_in_out[idx] = (fixed_t)((double)i_fb_in_out[idx] * factor);
        q_fb_in_out[idx] = (fixed_t)((double)q_fb_in_out[idx] * factor);
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

    // Allocate device memory for common arrays used across modes
    fixed_t *d_i_symbols, *d_q_symbols;
    fixed_t *d_i_psf, *d_q_psf;
    data_t *d_dpd_i, *d_dpd_q; // DPD output buffers on device
    data_ty *d_dac_i_arr, *d_dac_q_arr;
    data_t *d_qm_out_buf;
    sample_type *d_duc_out;
    data_t *d_amp_out_i, *d_amp_out_q, *d_amp_magnitude, *d_amp_gain_lin, *d_amp_gain_db;
    baseband_t *d_ddc_i_out, *d_ddc_q_out;
    adc_out_t *d_adc_i_out, *d_adc_q_out;
    fixed_t *d_i_psf_fb_device, *d_q_psf_fb_device; // Feedback PSF output on device

    CUDA_CHECK(cudaMalloc(&d_i_symbols, MAX_SYMBOLS * sizeof(fixed_t)));
    CUDA_CHECK(cudaMalloc(&d_q_symbols, MAX_SYMBOLS * sizeof(fixed_t)));
    CUDA_CHECK(cudaMalloc(&d_i_psf, DATA_LEN * sizeof(fixed_t)));
    CUDA_CHECK(cudaMalloc(&d_q_psf, DATA_LEN * sizeof(fixed_t)));
    CUDA_CHECK(cudaMalloc(&d_dpd_i, DATA_LEN * sizeof(data_t)));
    CUDA_CHECK(cudaMalloc(&d_dpd_q, DATA_LEN * sizeof(data_t)));
    CUDA_CHECK(cudaMalloc(&d_dac_i_arr, DATA_LEN * sizeof(data_ty)));
    CUDA_CHECK(cudaMalloc(&d_dac_q_arr, DATA_LEN * sizeof(data_ty)));
    CUDA_CHECK(cudaMalloc(&d_qm_out_buf, DATA_LEN * sizeof(data_t)));
    CUDA_CHECK(cudaMalloc(&d_duc_out, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(sample_type)));
    CUDA_CHECK(cudaMalloc(&d_amp_out_i, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
    CUDA_CHECK(cudaMalloc(&d_amp_out_q, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
    CUDA_CHECK(cudaMalloc(&d_amp_magnitude, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
    CUDA_CHECK(cudaMalloc(&d_amp_gain_lin, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));
    CUDA_CHECK(cudaMalloc(&d_amp_gain_db, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t)));

    const int ADC_LEN = (DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR;
    CUDA_CHECK(cudaMalloc(&d_ddc_i_out, ADC_LEN * sizeof(baseband_t)));
    CUDA_CHECK(cudaMalloc(&d_ddc_q_out, ADC_LEN * sizeof(baseband_t)));
    CUDA_CHECK(cudaMalloc(&d_adc_i_out, ADC_LEN * sizeof(adc_out_t)));
    CUDA_CHECK(cudaMalloc(&d_adc_q_out, ADC_LEN * sizeof(adc_out_t)));
    CUDA_CHECK(cudaMalloc(&d_i_psf_fb_device, ADC_LEN * sizeof(fixed_t)));
    CUDA_CHECK(cudaMalloc(&d_q_psf_fb_device, ADC_LEN * sizeof(fixed_t)));

    // Copy initial symbols to device
    CUDA_CHECK(cudaMemcpy(d_i_symbols, i_symbols, MAX_SYMBOLS * sizeof(fixed_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_symbols, q_symbols, MAX_SYMBOLS * sizeof(fixed_t), cudaMemcpyHostToDevice));

    // Pulse shaping (host-callable, but its internal functions are __host__ __device__)
    // For full GPU acceleration, this would be a kernel launch. For now, it copies to host, processes, then copies back.
    pulse_shape(i_symbols, q_symbols, i_psf, q_psf);
    CUDA_CHECK(cudaMemcpy(d_i_psf, i_psf, DATA_LEN * sizeof(fixed_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_psf, q_psf, DATA_LEN * sizeof(fixed_t), cudaMemcpyHostToDevice));

    uint8_t phase_inc_val = 2; // NCO phase increment for QM

    int threadsPerBlock = 256;
    int blocksPerGrid = (DATA_LEN + threadsPerBlock - 1) / threadsPerBlock;


    if (mode == BASELINE) {
        // ----------- BASELINE: DPD passthrough, output first pass only -----------
        // DPD passthrough: simply copy i_psf/q_psf to dpd_i/q
        baseline_dpd_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_i_psf, d_q_psf, d_dpd_i, d_dpd_q);
        CUDA_CHECK(cudaDeviceSynchronize());

        // DAC and QM
        dac_qm_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_dpd_i, d_dpd_q,
            d_dac_i_arr, d_dac_q_arr,
            d_qm_out_buf,
            phase_inc_val
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Digital Upconverter
        digital_upconverter(
            (const sample_type*)d_qm_out_buf,
            (const sample_type*)d_qm_out_buf, // Assuming q_in is 0 for single-channel in original DUC call
            d_duc_out,
            0x40000000, // freq_control_word
            1 // enable
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Amplifier (Saleh model)
        int amp_total_samples = DATA_LEN * INTERPOLATION_FACTOR;
        blocksPerGrid = (amp_total_samples + threadsPerBlock - 1) / threadsPerBlock;
        amplifier_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_duc_out,
            d_amp_out_i, d_amp_out_q, d_amp_magnitude, d_amp_gain_lin, d_amp_gain_db,
            amp_total_samples
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Digital Downconverter
        // ddc_demodulator handles its own internal device memory management
        ddc_demodulator(
            (const rf_sample_t*)d_amp_out_i, // Input is from amplifier output
            d_ddc_i_out,
            d_ddc_q_out,
            amp_total_samples, // num_samples for DDC is the total samples from DUC/AMP
            0x40000000, // ddc_freq_word
            25000.0f // ddc_gain
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // ADC
        // dual_adc_system handles its own internal device memory management
        dual_adc_system(
            (const adc_in_t*)d_ddc_i_out, (const adc_in_t*)d_ddc_q_out,
            d_adc_i_out, d_adc_q_out
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Pulse Shaping for Feedback (PSF_FB)
        // Copy ADC output to host for PSF, then copy PSF output to device for normalization
        fixed_t *h_i_adc_out = (fixed_t*)malloc(ADC_LEN * sizeof(fixed_t));
        fixed_t *h_q_adc_out = (fixed_t*)malloc(ADC_LEN * sizeof(fixed_t));
        CUDA_CHECK(cudaMemcpy(h_i_adc_out, d_adc_i_out, ADC_LEN * sizeof(adc_out_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_q_adc_out, d_adc_q_out, ADC_LEN * sizeof(adc_out_t), cudaMemcpyDeviceToHost));

        pulse_shape(h_i_adc_out, h_q_adc_out, i_psf_fb, q_psf_fb); // Host call for PSF
        free(h_i_adc_out);
        free(h_q_adc_out);

        CUDA_CHECK(cudaMemcpy(d_i_psf_fb_device, i_psf_fb, ADC_LEN * sizeof(fixed_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q_psf_fb_device, q_psf_fb, ADC_LEN * sizeof(fixed_t), cudaMemcpyHostToDevice));

        // Normalization of feedback path
        double ff_rms = 0, fb_rms = 0;
        // Copy d_i_psf to host for RMS calculation
        CUDA_CHECK(cudaMemcpy(i_psf, d_i_psf, DATA_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        // Use i_psf_fb (host) for fb_rms since it was just updated by pulse_shape
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
        normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_i_psf_fb_device, d_q_psf_fb_device, norm_factor, ADC_LEN);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy final results for baseline back to host for test_for_mde.cpp to write to files
        CUDA_CHECK(cudaMemcpy(i_psf_fb, d_i_psf_fb_device, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_psf_fb, d_q_psf_fb_device, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));

        // No need to copy other outputs back for BASELINE mode, as they are not written to files in test_for_mde.cpp for this mode.
        // If needed, add cudaMemcpy calls here.

        // Free all device memory (common cleanup at the end of circuit_final)
        // This will be done once at the very end of this function.
        // The memory is reused for ADAPT and FINAL modes.

        return;
    }

    if (mode == ADAPT) {
        // ----------- ADAPTATION: Indirect Learning -----------
        // This mode involves the `dpd_mde` (Differential Evolution) algorithm,
        // which currently runs on the host and manages its own static state.
        // It consumes `i_psf_fb` and `q_psf_fb` (from the previous run's feedback path)
        // and `i_psf`, `q_psf` (from the feedforward path) to update the `w` weights.
        // No new kernels are launched in this mode; it's a host-side computation loop.

        // Ensure i_psf, q_psf and i_psf_fb, q_psf_fb are on host for dpd_mde
        CUDA_CHECK(cudaMemcpy(i_psf, d_i_psf, DATA_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_psf, d_q_psf, DATA_LEN * sizeof(fixed_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(i_psf_fb, d_i_psf_fb_device, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_psf_fb, d_q_psf_fb_device, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));


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

        // The weights 'w' are updated by dpd_mde on the host.
        // No further device operations needed in ADAPT mode, as it's just updating coefficients.

        return;
    }

    if (mode == FINAL) {
        // ----------- FINAL: DPD with adapted weights, output second pass only -----------

        // Copy adapted DPD weights 'w' to device for the kernel.
        // A __constant__ memory approach would be more efficient if 'w' were truly constant per kernel launch.
        // For simplicity, we'll pass it as a parameter, or copy to a device global array.
        // For now, we'll assume `w` is accessible by the kernel because it's a static global.
        // If `w` were a local variable, it would need to be copied to device memory.

        // DPD application
        // This kernel applies the DPD polynomial using the current 'w'
        apply_dpd_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_i_psf, d_q_psf, // Input to DPD is pulse-shaped signal
            d_dpd_i, d_dpd_q, // Output of DPD
            w, // Pass the adapted weights 'w'
            DATA_LEN
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // DAC and QM
        dac_qm_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_dpd_i, d_dpd_q, // Input is DPD output
            d_dac_i_arr, d_dac_q_arr,
            d_qm_out_buf,
            phase_inc_val
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Digital Upconverter
        digital_upconverter(
            (const sample_type*)d_qm_out_buf,
            (const sample_type*)d_qm_out_buf, // Assuming q_in is 0
            d_duc_out,
            0x40000000, // freq_control_word
            1 // enable
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Amplifier (Saleh model)
        int amp_total_samples = DATA_LEN * INTERPOLATION_FACTOR;
        blocksPerGrid = (amp_total_samples + threadsPerBlock - 1) / threadsPerBlock;
        amplifier_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_duc_out,
            d_amp_out_i, d_amp_out_q, d_amp_magnitude, d_amp_gain_lin, d_amp_gain_db,
            amp_total_samples
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Digital Downconverter
        ddc_demodulator(
            (const rf_sample_t*)d_amp_out_i,
            d_ddc_i_out,
            d_ddc_q_out,
            amp_total_samples,
            0x40000000,
            25000.0f
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // ADC
        dual_adc_system(
            (const adc_in_t*)d_ddc_i_out, (const adc_in_t*)d_ddc_q_out,
            d_adc_i_out, d_adc_q_out
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Pulse Shaping for Feedback (PSF_FB)
        // Copy ADC output to host for PSF, then copy PSF output to device for normalization
        fixed_t *h_i_adc_out = (fixed_t*)malloc(ADC_LEN * sizeof(fixed_t));
        fixed_t *h_q_adc_out = (fixed_t*)malloc(ADC_LEN * sizeof(fixed_t));
        CUDA_CHECK(cudaMemcpy(h_i_adc_out, d_adc_i_out, ADC_LEN * sizeof(adc_out_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_q_adc_out, d_adc_q_out, ADC_LEN * sizeof(adc_out_t), cudaMemcpyDeviceToHost));

        pulse_shape(h_i_adc_out, h_q_adc_out, i_psf_fb, q_psf_fb); // Host call for PSF
        free(h_i_adc_out);
        free(h_q_adc_out);

        CUDA_CHECK(cudaMemcpy(d_i_psf_fb_device, i_psf_fb, ADC_LEN * sizeof(fixed_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q_psf_fb_device, q_psf_fb, ADC_LEN * sizeof(fixed_t), cudaMemcpyHostToDevice));

        // Normalization of feedback path
        double ff_rms = 0, fb_rms = 0;
        // Copy d_i_psf to host for RMS calculation
        CUDA_CHECK(cudaMemcpy(i_psf, d_i_psf, DATA_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        // Use i_psf_fb (host) for fb_rms since it was just updated by pulse_shape
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
        normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_i_psf_fb_device, d_q_psf_fb_device, norm_factor, ADC_LEN);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy all relevant outputs back to host for test_for_mde.cpp to write to files
        CUDA_CHECK(cudaMemcpy(dpd_i, d_dpd_i, DATA_LEN * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dpd_q, d_dpd_q, DATA_LEN * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dac_i_arr, d_dac_i_arr, DATA_LEN * sizeof(data_ty), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dac_q_arr, d_dac_q_arr, DATA_LEN * sizeof(data_ty), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qm_out_buf, d_qm_out_buf, DATA_LEN * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(duc_out, d_duc_out, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(sample_type), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amp_out_i, d_amp_out_i, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amp_out_q, d_amp_out_q, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amp_magnitude, d_amp_magnitude, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amp_gain_lin, d_amp_gain_lin, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(amp_gain_db, d_amp_gain_db, (DATA_LEN * INTERPOLATION_FACTOR) * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ddc_i_out, d_ddc_i_out, ADC_LEN * sizeof(baseband_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ddc_q_out, d_ddc_q_out, ADC_LEN * sizeof(baseband_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(adc_i_out, d_adc_i_out, ADC_LEN * sizeof(adc_out_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(adc_q_out, d_adc_q_out, ADC_LEN * sizeof(adc_out_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(i_psf_fb, d_i_psf_fb_device, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_psf_fb, d_q_psf_fb_device, ADC_LEN * sizeof(fixed_t), cudaMemcpyDeviceToHost));

        return;
    }

    // Free all common device memory at the end of circuit_final execution
    cudaFree(d_i_symbols); cudaFree(d_q_symbols);
    cudaFree(d_i_psf); cudaFree(d_q_psf);
    cudaFree(d_dpd_i); cudaFree(d_dpd_q);
    cudaFree(d_dac_i_arr); cudaFree(d_dac_q_arr); cudaFree(d_qm_out_buf);
    cudaFree(d_duc_out);
    cudaFree(d_amp_out_i); cudaFree(d_amp_out_q); cudaFree(d_amp_magnitude); cudaFree(d_amp_gain_lin); cudaFree(d_amp_gain_db);
    cudaFree(d_ddc_i_out); cudaFree(d_ddc_q_out);
    cudaFree(d_adc_i_out); cudaFree(d_adc_q_out);
    cudaFree(d_i_psf_fb_device); cudaFree(d_q_psf_fb_device);
}
