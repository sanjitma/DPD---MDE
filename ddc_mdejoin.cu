// CUDA port: Remove HLS/Vivado includes
#include <stdint.h>
#include <cmath>
// Configuration parameters
#define DECIM_FACTOR 8            // Decimation factor
#define FIR_TAPS 63               // Number of FIR filter taps

// CUDA-compatible types
typedef float rf_sample_t;       // 16-bit RF input samples
typedef float filter_coeff_t;    // Filter coefficient type
typedef float filter_accum_t;    // Accumulator with growth for FIR filters
typedef float baseband_t;        // Baseband I/Q output samples
typedef uint16_t phase_t;        // NCO phase accumulator index

/**
 * Digital Downconverter (DDC) and Demodulator
 *
 * @param rf_in       - RF input signal
 * @param i_out       - I channel baseband output
 * @param q_out       - Q channel baseband output
 * @param num_samples - Number of input samples to process
 * @param freq_word   - NCO frequency control word
 * @param gain        - Output gain control
 */
// CUDA kernel for DDC (simplified, FIR and NCO logic can be ported as needed)
__global__ void ddc_demodulator_kernel(
    const rf_sample_t *rf_in,
    baseband_t *i_out,
    baseband_t *q_out,
    int num_samples,
    uint32_t freq_word,
    float gain
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        // TODO: Port FIR, NCO, and mixing logic here for full CUDA compatibility
        // For demonstration, just copy input to output
        i_out[idx] = rf_in[idx] * gain;
        q_out[idx] = 0.0f;
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
    cudaMalloc(&d_i_out, num_samples * sizeof(baseband_t));
    cudaMalloc(&d_q_out, num_samples * sizeof(baseband_t));
    cudaMemcpy(d_rf_in, rf_in, num_samples * sizeof(rf_sample_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_samples + threadsPerBlock - 1) / threadsPerBlock;
    ddc_demodulator_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_rf_in, d_i_out, d_q_out, num_samples, freq_word, gain);
    cudaDeviceSynchronize();

    cudaMemcpy(i_out, d_i_out, num_samples * sizeof(baseband_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(q_out, d_q_out, num_samples * sizeof(baseband_t), cudaMemcpyDeviceToHost);

    cudaFree(d_rf_in);
    cudaFree(d_i_out);
    cudaFree(d_q_out);
}

// Assume these headers define: extern const filter_coeff_t sin_lut[1024]; etc.

    // Filter state buffers for I and Q channels
    static filter_accum_t i_buffer[FIR_TAPS];
    static filter_accum_t q_buffer[FIR_TAPS];
#pragma HLS ARRAY_PARTITION variable=i_buffer cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=q_buffer cyclic factor=8

    // Initialize filter buffers
    for (int i = 0; i < FIR_TAPS; i++) {
#pragma HLS UNROLL
        i_buffer[i] = 0;
        q_buffer[i] = 0;
    }

    // NCO phase accumulator
    ap_uint<32> phase_acc = 0;

    // Track output sample count
    int out_sample_idx = 0;

    // Decimation counter
    int decim_count = 0;

    // Main processing loop
    for (int n = 0; n < num_samples; n++) {
#pragma HLS LOOP_TRIPCOUNT min=1024 max=16384 avg=8192
#pragma HLS PIPELINE II=1

        // Get input sample
        rf_sample_t rf_sample = rf_in[n];
        if (n < 50) std::cout << "[n=" << n << "] rf_sample = " << rf_sample.to_double() << std::endl;

        // Update NCO phase
        phase_acc += freq_word;

        // Extract the most significant bits for table lookup
        phase_t table_idx = phase_acc >> (32 - 10);  // Use top 10 bits (log2(NCO_LUT_SIZE))

        // Look up sine and cosine values for quadrature mixing
        filter_coeff_t sin_val = sine_lut[table_idx];
        filter_coeff_t cos_val = cosine_lut[table_idx];
        if (n < 50) std::cout << "[n=" << n << "] sin_val = " << sin_val.to_double() << ", cos_val = " << cos_val.to_double() << std::endl;

        // Quadrature mixing (downconversion)
        filter_accum_t i_mixed = rf_sample * cos_val;
        filter_accum_t q_mixed = rf_sample * -sin_val;  // Negative for downconversion

        if (n < 50) std::cout << "[n=" << n << "] i_mixed = " << i_mixed.to_double() << ", q_mixed = " << q_mixed.to_double() << std::endl;

        // Shift samples through delay line
        for (int i = FIR_TAPS-1; i > 0; i--) {
#pragma HLS UNROLL factor=8
            i_buffer[i] = i_buffer[i-1];
            q_buffer[i] = q_buffer[i-1];
        }

        // Insert new mixed samples
        i_buffer[0] = i_mixed;
        q_buffer[0] = q_mixed;

        // Apply decimating filter
        if (decim_count == 0) {
            // Compute FIR filter outputs
            filter_accum_t i_acc = 0;
            filter_accum_t q_acc = 0;

            for (int i = 0; i < FIR_TAPS; i++) {
#pragma HLS UNROLL factor=8
                i_acc += i_buffer[i] * lpf_coeffs[i];
                q_acc += q_buffer[i] * lpf_coeffs[i];
            }

            if (out_sample_idx < 50) std::cout << "[out_sample_idx=" << out_sample_idx << "] i_acc = " << i_acc.to_double() << ", q_acc = " << q_acc.to_double() << std::endl;

            // Apply gain scaling and store results
            i_out[out_sample_idx] = i_acc * gain;
            q_out[out_sample_idx] = q_acc * gain;

            if (out_sample_idx < 50) std::cout << "[out_sample_idx=" << out_sample_idx << "] i_acc = " << i_acc.to_double() << ", q_acc = " << q_acc.to_double() << std::endl;


            // Increment output sample counter
            out_sample_idx++;
        }

        // Update decimation counter
        decim_count = (decim_count + 1) % DECIM_FACTOR;
    }
}
