#include <cmath>
#include <cstdint>
#include "duc_mdejoiner.h"

// Configuration parameters
#define INTERPOLATION_FACTOR 8    // Interpolation/upsampling factor
#define NCO_LUT_SIZE 1024         // Size of sine/cosine lookup table
#define NCO_LUT_BITS 10           // log2(NCO_LUT_SIZE)
#define NUM_FILTER_TAPS 63        // FIR filter taps (odd number)

// Fixed-point type definitions
typedef ap_fixed<24,8> sample_type;      // 16-bit sample: 1 sign bit, 1 integer bit, 14 fractional bits
typedef ap_fixed<24,12> coeff_type;       // Filter coefficient type
typedef ap_fixed<24,12> acc_type;         // Accumulator with extra bits for growth
typedef ap_uint<12> phase_type;          // Phase accumulator type

// Filter coefficients - Polyphase lowpass (pre-computed)
const coeff_type FILTER_COEFFS[NUM_FILTER_TAPS] = {
    // These are example LPF coefficients, replace with your actual filter design
    0.00156, 0.00258, 0.00376, 0.00511, 0.00662, 0.00828, 0.01007, 0.01196, 0.01392,
    0.01594, 0.01796, 0.01997, 0.02193, 0.02379, 0.02553, 0.02713, 0.02855, 0.02978,
    0.03079, 0.03157, 0.03209, 0.03235, 0.03235, 0.03209, 0.03157, 0.03079, 0.02978,
    0.02855, 0.02713, 0.02553, 0.02379, 0.02193, 0.01997, 0.01796, 0.01594, 0.01392,
    0.01196, 0.01007, 0.00828, 0.00662, 0.00511, 0.00376, 0.00258, 0.00156, 0.00069,
    0.00000, -0.00053, -0.00091, -0.00113, -0.00123, -0.00122, -0.00113, -0.00098,
    -0.00079, -0.00058, -0.00038, -0.00020, -0.00006, 0.00003, 0.00009, 0.00010, 0.00009,
    0.00006
};

// Sine/Cosine lookup table (pre-computed)
const sample_type sine_lut[NCO_LUT_SIZE] = {
    #include "sin_lut.h"  // Should contain 1024 values
};
const sample_type cosine_lut[NCO_LUT_SIZE] = {
    #include "cos_lut.h"  // Should contain 1024 values
};

// Simple FIR filter for interpolation (not true polyphase)
static void fir_filter(
    sample_type shift_reg[NUM_FILTER_TAPS],
    sample_type input_sample,
    sample_type &output_sample
) {
    // Shift the register
    for (int i = NUM_FILTER_TAPS-1; i > 0; i--) {
        shift_reg[i] = shift_reg[i-1];
    }
    shift_reg[0] = input_sample;

    // Calculate filter output
    acc_type acc = 0;
    for (int i = 0; i < NUM_FILTER_TAPS; i++) {
        acc += shift_reg[i] * FILTER_COEFFS[i];
    }
    output_sample = acc;
}

// Array-based digital upconverter
void digital_upconverter(
    const sample_type *i_in,
    const sample_type *q_in,
    sample_type *signal_out,
    uint32_t freq_control_word,
    uint8_t enable
) {
    if (!enable) return;

    sample_type i_shift_reg[NUM_FILTER_TAPS] = {0};
    sample_type q_shift_reg[NUM_FILTER_TAPS] = {0};
    uint32_t phase_acc = 0;

    // Assume input length is DATA_LEN (8192) for now
    const int INPUT_LEN = 8192;
    int out_idx = 0;
    for (int n = 0; n < INPUT_LEN; ++n) {
        sample_type i_sample = i_in[n];
        sample_type q_sample = q_in[n];
        for (int phase = 0; phase < INTERPOLATION_FACTOR; ++phase) {
            sample_type i_fir_in = (phase == 0) ? i_sample : sample_type(0);
            sample_type q_fir_in = (phase == 0) ? q_sample : sample_type(0);
            sample_type i_interp, q_interp;
            fir_filter(i_shift_reg, i_fir_in, i_interp);
            fir_filter(q_shift_reg, q_fir_in, q_interp);
            phase_acc += freq_control_word;
            uint32_t lut_addr = (phase_acc >> (32 - NCO_LUT_BITS)) & (NCO_LUT_SIZE-1);
            sample_type sin_val = sine_lut[lut_addr];
            sample_type cos_val = cosine_lut[lut_addr];
            sample_type upconverted = i_interp * cos_val - q_interp * sin_val;
            signal_out[out_idx++] = upconverted;
        }
    }
}
