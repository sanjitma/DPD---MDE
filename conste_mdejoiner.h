
#include <stdint.h>

typedef float fixed_point_t;
#define MAX_INPUT_BYTES 8192
#define MAX_SYMBOLS 32768
typedef int modulation_type;

void conste(
    const float input_bytes[MAX_INPUT_BYTES],
    int num_bits,
    fixed_point_t output_symbols_I[MAX_SYMBOLS],
    fixed_point_t output_symbols_Q[MAX_SYMBOLS]
);
