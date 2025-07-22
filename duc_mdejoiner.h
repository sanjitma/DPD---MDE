#ifndef DIGITALUC_H
#define DIGITALUC_H



// Configuration parameters (should match your .cpp)
#define INTERPOLATION_FACTOR 8
#define NCO_LUT_SIZE 1024
#define NCO_LUT_BITS 10
#define NUM_FILTER_TAPS 63

typedef float sample_type;
typedef float coeff_type;
typedef float acc_type;
typedef uint16_t phase_type;

// Top-level DUC function prototype
void digital_upconverter(
    const sample_type *i_in,
    const sample_type *q_in,
    sample_type *signal_out,
    uint32_t freq_control_word,
    uint8_t enable
);

#endif // DIGITALUC_H
