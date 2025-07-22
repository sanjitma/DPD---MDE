#ifndef DDC_JOINER_H
#define DDC_JOINER_H

#include <stdint.h>

#define DECIM_FACTOR 8
#define FIR_TAPS 63

typedef float rf_sample_t;       // 16-bit RF input samples
typedef float filter_coeff_t;    // Filter coefficient type
typedef float filter_accum_t;    // Accumulator with growth for FIR filters
typedef float baseband_t;        // Baseband I/Q output samples

void ddc_demodulator(
    const rf_sample_t rf_in[],
    baseband_t i_out[],
    baseband_t q_out[],
    int num_samples,
    uint32_t freq_word,
    float gain
);

#endif // DDC_JOINER_H
