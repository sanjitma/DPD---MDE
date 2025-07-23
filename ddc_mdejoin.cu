#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>
#include <cstdio> // For printf in debug, if needed
#include "luts.cuh"

// Configuration parameters
#define DECIM_FACTOR 8            // Decimation factor
#define FIR_TAPS 63               // Number of FIR filter taps
#define NCO_LUT_SIZE 1024         // Size of sine/cosine lookup table
#define NCO_LUT_BITS 10           // log2(NCO_LUT_SIZE)

// CUDA-compatible types
typedef float rf_sample_t;       // RF input samples
typedef float filter_coeff_t;    // Filter coefficient type
typedef float filter_accum_t;    // Accumulator with growth for FIR filters
typedef float baseband_t;        // Baseband I/Q output samples
typedef uint32_t phase_t;        // NCO phase accumulator index (changed to uint32_t for full frequency word precision)

// Define LPF coefficients as a constant array in device memory
__constant__ filter_coeff_t d_lpf_coeffs[FIR_TAPS] = {
    0.00156f, 0.00258f, 0.00376f, 0.00511f, 0.00662f, 0.00828f, 0.01007f, 0.01196f, 0.01392f,
    0.01594f, 0.01796f, 0.01997f, 0.02193f, 0.02379f, 0.02553f, 0.02713f, 0.02855f, 0.02978f,
    0.03079f, 0.03157f, 0.03209f, 0.03235f, 0.03235f, 0.03209f, 0.03157f, 0.03079f, 0.02978f,
    0.02855f, 0.02713f, 0.02553f, 0.02379f, 0.02193f, 0.01997f, 0.01796f, 0.01594f, 0.01392f,
    0.01196f, 0.01007f, 0.00828f, 0.00662f, 0.00511f, 0.00376f, 0.00258f, 0.00156f, 0.00069f,
    0.00000f, -0.00053f, -0.00091f, -0.00113f, -0.00123f, -0.00122f, -0.00113f, -0.00098f,
    -0.00079f, -0.00058f, -0.00038f, -0.00020f, -0.00006f, 0.00003f, 0.00009f, 0.00010f, 0.00009f,
    0.00006f
}; //

/**
 * Digital Downconverter (DDC) and Demodulator CUDA kernel
 *
 * This kernel processes input samples, performs quadrature mixing,
 * FIR filtering, and decimation. Each thread processes a chunk of samples.
 * The FIR filter state and NCO phase are managed per thread,
 * effectively serializing the FIR and decimation per conceptual channel,
 * but allowing parallel processing of different output samples.
 *
 * @param rf_in       - RF input signal (device pointer)
 * @param i_out       - I channel baseband output (device pointer)
 * @param q_out       - Q channel baseband output (device pointer)
 * @param num_samples - Total number of input samples to process
 * @param freq_word   - NCO frequency control word
 * @param gain        - Output gain control
 */
__global__ void ddc_demodulator_kernel(
    const rf_sample_t *rf_in,
    baseband_t *i_out,
    baseband_t *q_out,
    int num_samples,
    uint32_t freq_word,
    float gain
) {
    // Each thread processes one output sample, requiring it to process DECIM_FACTOR input samples.
    // The total number of input samples is num_samples.
    // The total number of output samples is num_samples / DECIM_FACTOR.
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start_in_idx = out_idx * DECIM_FACTOR;

    if (start_in_idx < num_samples) {
        // Local filter state buffers for I and Q channels for this thread
        filter_accum_t i_buffer[FIR_TAPS] = {0};
        filter_accum_t q_buffer[FIR_TAPS] = {0};

        // NCO phase accumulator for this thread
        // Initialize based on the starting input sample to maintain phase continuity
        // Simplified: assuming each block starts fresh or phase is handled externally.
        // For truly continuous phase across blocks/threads, phase accumulation needs
        // to be synchronized or pre-calculated for each thread's starting point.
        // For this port, it's a direct translation for local operation per output group.
        phase_t phase_acc = 0; // Each thread processes its own set of samples

        // Loop over the input samples for this output sample
        for (int i_sub_idx = 0; i_sub_idx < DECIM_FACTOR; ++i_sub_idx) {
            int current_in_idx = start_in_idx + i_sub_idx;
            if (current_in_idx >= num_samples) {
                break; // Ensure we don't go out of bounds for the input array
            }

            rf_sample_t rf_sample = rf_in[current_in_idx];

            // Update NCO phase
            phase_acc += freq_word;

            // Extract the most significant bits for table lookup
            // Use top NCO_LUT_BITS for index
            uint32_t table_idx = (phase_acc >> (32 - NCO_LUT_BITS)) & (NCO_LUT_SIZE - 1);

            // Look up sine and cosine values for quadrature mixing from device constant memory
            filter_coeff_t sin_val = d_sine_lut[table_idx];
            filter_coeff_t cos_val = d_cosine_lut[table_idx];

            // Quadrature mixing (downconversion)
            filter_accum_t i_mixed = rf_sample * cos_val;
            filter_accum_t q_mixed = rf_sample * -sin_val; // Negative for downconversion

            // Shift samples through delay line for FIR filter
            for (int i = FIR_TAPS - 1; i > 0; --i) {
                i_buffer[i] = i_buffer[i - 1];
                q_buffer[i] = q_buffer[i - 1];
            }

            // Insert new mixed samples
            i_buffer[0] = i_mixed;
            q_buffer[0] = q_mixed;
        } // End of DECIM_FACTOR samples loop

        // Compute FIR filter outputs (only for the decimated sample)
        filter_accum_t i_acc = 0;
        filter_accum_t q_acc = 0;

        for (int i = 0; i < FIR_TAPS; ++i) {
            i_acc += i_buffer[i] * d_lpf_coeffs[i];
            q_acc += q_buffer[i] * d_lpf_coeffs[i];
        }

        // Apply gain scaling and store results
        i_out[out_idx] = i_acc * gain;
        q_out[out_idx] = q_acc * gain;
    }
}

// Host wrapper for DDC
void ddc_demodulator(
    const rf_sample_t rf_in[],
    baseband_t i_out[],
    baseband_t q_out[],
    int num_samples,
    uint32_t freq_word,
    float gain
) {
    rf_sample_t *d_rf_in;
    baseband_t *d_i_out, *d_q_out;
    cudaMalloc(&d_rf_in, num_samples * sizeof(rf_sample_t));
    // Output array size is num_samples / DECIM_FACTOR
    int num_output_samples = num_samples / DECIM_FACTOR;
    cudaMalloc(&d_i_out, num_output_samples * sizeof(baseband_t));
    cudaMalloc(&d_q_out, num_output_samples * sizeof(baseband_t));
    cudaMemcpy(d_rf_in, rf_in, num_samples * sizeof(rf_sample_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    // Calculate blocks per grid based on output samples
    int blocksPerGrid = (num_output_samples + threadsPerBlock - 1) / threadsPerBlock;

    ddc_demodulator_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_rf_in, d_i_out, d_q_out, num_samples, freq_word, gain);
    cudaDeviceSynchronize();

    cudaMemcpy(i_out, d_i_out, num_output_samples * sizeof(baseband_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(q_out, d_q_out, num_output_samples * sizeof(baseband_t), cudaMemcpyDeviceToHost);

    cudaFree(d_rf_in);
    cudaFree(d_i_out);
    cudaFree(d_q_out);
}
