#include <stdint.h>
#include <iostream>
#include <cmath>
#include "psf_mdejoiner.h"
#include "mde.h"
#include "dac_mdejoiner.h"
#include "qm_mdejoiner.h"
#include "duc_mdejoiner.h"
#include "pa_mdejoiner.h"
#include "ddc_mdejoiner.h"
#include <cuda_runtime.h>
// Remove HLS includes
// #include "hls_stream.h"
// #include <ap_fixed.h>
// #include <ap_int.h>
#include "adc_mdejoiner.h"
// Removed SYCL include and namespace for CUDA compatibility

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
typedef float adc_out_t;
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

// MDE DPD weights
static ccoef_t w[K][MEMORY_DEPTH] = {0};
static bool weights_initialized = false;

// CUDA kernel for baseline mode (minimal stub, real implementation needed)
__global__ void baseline_kernel(
    const data_t *i_psf, const data_t *q_psf,
    data_t *dpd_i, data_t *dpd_q,
    data_ty *dac_i_arr, data_ty *dac_q_arr,
    data_t *qm_out_buf
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < DATA_LEN) {
        // Minimal passthrough: just copy i_psf/q_psf to dpd_i/dpd_q, zero others
        dpd_i[idx] = i_psf[idx];
        dpd_q[idx] = q_psf[idx];
        dac_i_arr[idx] = 0;
        dac_q_arr[idx] = 0;
        qm_out_buf[idx] = 0;
        // TODO: Implement real kernel logic as needed
    }
}

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
    // CUDA porting: Remove SYCL and HLS streams/pragmas
    // Use arrays and CUDA kernels for GPU implementation

    if (!weights_initialized) {
        for (int m = 0; m < MEMORY_DEPTH; ++m) {
            w[0][m].real = 1.0;
            w[0][m].imag = 0.0;
        }
        weights_initialized = true;
    }

    pulse_shape(i_symbols, q_symbols, i_psf, q_psf);

    uint8_t phase = 0;
    uint8_t phase_inc = 2;

    const int ADC_LEN = (DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR;


    if (mode == BASELINE) {
        // ----------- BASELINE: DPD passthrough, output first pass only -----------
        // CUDA kernel for main loop
        // Allocate device memory
        data_t *d_i_psf, *d_q_psf, *d_dpd_i, *d_dpd_q, *d_dac_i_arr, *d_dac_q_arr, *d_qm_out_buf;
        CUDA_CHECK(cudaMalloc(&d_i_psf, DATA_LEN * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_q_psf, DATA_LEN * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_dpd_i, DATA_LEN * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_dpd_q, DATA_LEN * sizeof(data_t)));
        CUDA_CHECK(cudaMalloc(&d_dac_i_arr, DATA_LEN * sizeof(data_ty)));
        CUDA_CHECK(cudaMalloc(&d_dac_q_arr, DATA_LEN * sizeof(data_ty)));
        CUDA_CHECK(cudaMalloc(&d_qm_out_buf, DATA_LEN * sizeof(data_t)));
        CUDA_CHECK(cudaMemcpy(d_i_psf, i_psf, DATA_LEN * sizeof(data_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q_psf, q_psf, DATA_LEN * sizeof(data_t), cudaMemcpyHostToDevice));

        // Define CUDA kernel
        // You should move this to a .cu file for real builds
        // __global__ function must be outside host function in real CUDA code
        // For demonstration, you can place this in a .cu file

        // See below for kernel definition

        int threadsPerBlock = 256;
        int blocksPerGrid = (DATA_LEN + threadsPerBlock - 1) / threadsPerBlock;
        baseline_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_i_psf, d_q_psf, d_dpd_i, d_dpd_q, d_dac_i_arr, d_dac_q_arr, d_qm_out_buf);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(dpd_i, d_dpd_i, DATA_LEN * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dpd_q, d_dpd_q, DATA_LEN * sizeof(data_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dac_i_arr, d_dac_i_arr, DATA_LEN * sizeof(data_ty), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dac_q_arr, d_dac_q_arr, DATA_LEN * sizeof(data_ty), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qm_out_buf, d_qm_out_buf, DATA_LEN * sizeof(data_t), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_i_psf));
        CUDA_CHECK(cudaFree(d_q_psf));
        CUDA_CHECK(cudaFree(d_dpd_i));
        CUDA_CHECK(cudaFree(d_dpd_q));
        CUDA_CHECK(cudaFree(d_dac_i_arr));
        CUDA_CHECK(cudaFree(d_dac_q_arr));
        CUDA_CHECK(cudaFree(d_qm_out_buf));

        // ...existing code...
        for (int i = 0; i < DATA_LEN * INTERPOLATION_FACTOR; ++i) {
            duc_out[i] = duc_out[i] * sample_type(300.0);
        }

        for (int i = 0; i < DATA_LEN * INTERPOLATION_FACTOR; ++i) {
            data_t local_amp_i, local_amp_q, local_amp_mag, local_amp_gain_lin, local_amp_gain_db;
            saleh_amplifier(
                data_t(duc_out[i]),
                data_t(0),
                local_amp_i, local_amp_q, local_amp_mag, local_amp_gain_lin, local_amp_gain_db
            );
            amp_out_i[i] = local_amp_i;
            amp_out_q[i] = local_amp_q;
            amp_magnitude[i] = local_amp_mag;
            amp_gain_lin[i] = local_amp_gain_lin;
            amp_gain_db[i] = local_amp_gain_db;
        }

        // ...existing code...
        return;
    }

    if (mode == ADAPT) {
        // ----------- ADAPTATION: Indirect Learning -----------
        for (int n = 0; n < DATA_LEN && n < ADC_LEN; ++n) {
            data_t i_in[MEMORY_DEPTH] = {0};
            data_t q_in[MEMORY_DEPTH] = {0};
            int fb_idx = n;
            for (int m = 0; m < MEMORY_DEPTH; ++m) {
                int idx = fb_idx - m;
                i_in[m] = data_t(idx >= 0) ? data_t(i_psf_fb[idx]) : data_t(0);
                q_in[m] = data_t(idx >= 0) ? data_t(q_psf_fb[idx]) : data_t(0);
            }
            int ref_idx = (n >= DELAY_OFFSET) ? (n - DELAY_OFFSET) : 0;
            data_t i_ref = (ref_idx < DATA_LEN) ? data_t(i_psf[ref_idx]) : data_t(0);
            data_t q_ref = (ref_idx < DATA_LEN) ? data_t(q_psf[ref_idx]) : data_t(0);

            data_t z_i, z_q;
            dpd_mde(i_in, q_in, i_ref, q_ref, w, &z_i, &z_q);
        }
        return;
    }

    if (mode == FINAL) {
        // ----------- FINAL: DPD with adapted weights, output second pass only -----------
        for (int n = 0; n < DATA_LEN; ++n) {
            data_t i_in[MEMORY_DEPTH] = {0};
            data_t q_in[MEMORY_DEPTH] = {0};
            for (int m = 0; m < MEMORY_DEPTH; ++m) {
                int idx = n - m;
                i_in[m] = data_t(idx >= 0) ? data_t(i_psf[idx]) : data_t(0);
                q_in[m] = data_t(idx >= 0) ? data_t(q_psf[idx]) : data_t(0);
            }
            data_t i_ref = i_psf[n];
            data_t q_ref = q_psf[n];

            data_t z_i, z_q;
            dpd_mde(i_in, q_in, i_ref, q_ref, w, &z_i, &z_q);
            dpd_i[n] = z_i;
            dpd_q[n] = z_q;

            data_ty dac_i, dac_q;
            dac_multibit_with_select(z_i, dac_i, 0);
            dac_multibit_with_select(z_q, dac_q, 1);
            dac_i_arr[n] = dac_i;
            dac_q_arr[n] = dac_q;

            data_t i_mod_fixed = data_t(dac_i) * data_t(1.0/128.0);
            data_t q_mod_fixed = data_t(dac_q) * data_t(1.0/128.0);

            data_t cos_lo, sin_lo;
            nco(phase, phase_inc, cos_lo, sin_lo);

            data_t qm_out = digital_qm(i_mod_fixed, q_mod_fixed, cos_lo, sin_lo);
            qm_out_buf[n] = qm_out;
        }

        // Remove HLS streams and pragmas, use arrays instead
        // hls::stream<sample_type> i_in_stream2, q_in_stream2, duc_out_stream2;
        // #pragma HLS STREAM variable=i_in_stream2 depth=64
        // #pragma HLS STREAM variable=q_in_stream2 depth=64
        // #pragma HLS STREAM variable=duc_out_stream2 depth=64
        for (int i = 0; i < DATA_LEN; ++i) {
            // i_in_stream2 << sample_type(qm_out_buf[i]); // Original line commented out
            // q_in_stream2 << sample_type(0); // Original line commented out
        }
        // ap_uint<32> freq_control_word = 0x40000000; // Original line commented out
        // ap_uint<1> enable = 1; // Original line commented out
        for (int i = 0; i < DATA_LEN; ++i) {
            // digital_upconverter(i_in_stream2, q_in_stream2, duc_out_stream2, freq_control_word, enable); // Original line commented out
        }
        int idx = 0;
        while (idx < DATA_LEN * INTERPOLATION_FACTOR) {
            sample_type val;
            // duc_out_stream2 >> val; // Original line commented out
            duc_out[idx++] = val;
        }
        for (int i = 0; i < DATA_LEN * INTERPOLATION_FACTOR; ++i) {
            duc_out[i] = duc_out[i] * sample_type(300.0);
        }

        for (int i = 0; i < DATA_LEN * INTERPOLATION_FACTOR; ++i) {
            data_t local_amp_i, local_amp_q, local_amp_mag, local_amp_gain_lin, local_amp_gain_db;
            saleh_amplifier(
                data_t(duc_out[i]),
                data_t(0),
                local_amp_i, local_amp_q, local_amp_mag, local_amp_gain_lin, local_amp_gain_db
            );
            amp_out_i[i] = local_amp_i;
            amp_out_q[i] = local_amp_q;
            amp_magnitude[i] = local_amp_mag;
            amp_gain_lin[i] = local_amp_gain_lin;
            amp_gain_db[i] = local_amp_gain_db;
        }

        static rf_sample_t ddc_in[DATA_LEN * INTERPOLATION_FACTOR];
        for (int i = 0; i < DATA_LEN * INTERPOLATION_FACTOR; ++i) {
            ddc_in[i] = rf_sample_t(amp_out_i[i]);
        }
        // ap_uint<32> ddc_freq_word = 0x40000000; // Original line commented out
        // ap_fixed<16,8> ddc_gain = 25000.0; // Original line commented out
        ddc_demodulator(
            ddc_in,
            ddc_i_out,
            ddc_q_out,
            DATA_LEN * INTERPOLATION_FACTOR,
            // ddc_freq_word, // Original line commented out
            // ddc_gain // Original line commented out
        );

        static adc_in_t adc_i_in[ADC_LEN], adc_q_in[ADC_LEN];
        for (int i = 0; i < ADC_LEN; ++i) {
            adc_i_in[i] = adc_in_t(ddc_i_out[i]);
            adc_q_in[i] = adc_in_t(ddc_q_out[i]);
        }
        dual_adc_system(adc_i_in, adc_q_in, adc_i_out, adc_q_out);

        static fixed_t i_psf_fb_in[ADC_LEN], q_psf_fb_in[ADC_LEN];
        for (int i = 0; i < ADC_LEN; ++i) {
            i_psf_fb_in[i] = fixed_t(adc_i_out[i]);
            q_psf_fb_in[i] = fixed_t(adc_q_out[i]);
        }
        pulse_shape(i_psf_fb_in, q_psf_fb_in, i_psf_fb, q_psf_fb);

        double ff_rms = 0, fb_rms = 0;
        for (int i = 0; i < DATA_LEN; ++i) {
            ff_rms += double(i_psf[i]) * double(i_psf[i]);
        }
        for (int i = 0; i < ADC_LEN; ++i) {
            fb_rms += double(i_psf_fb[i]) * double(i_psf_fb[i]);
        }
        ff_rms = std::sqrt(ff_rms / DATA_LEN);
        fb_rms = std::sqrt(fb_rms / ADC_LEN);

        double norm_factor = (fb_rms > 1e-12) ? (ff_rms / fb_rms) : 1.0;

        for (int i = 0; i < ADC_LEN; ++i) {
            i_psf_fb[i] = fixed_t(double(i_psf_fb[i]) * norm_factor);
            q_psf_fb[i] = fixed_t(double(q_psf_fb[i]) * norm_factor);
        }
        return;
    }
}
