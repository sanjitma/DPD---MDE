#include <stdint.h>
#include <cmath>

typedef float fixed_point_t;
#define MAX_INPUT_BYTES 8192
#define MAX_SYMBOLS 32768
// typedef int modulation_type; // Not used

// Added __host__ to indicate it's primarily a host function in this context
__host__ void conste(
    const float input_bytes[MAX_INPUT_BYTES],
    int num_bits,
    fixed_point_t output_symbols_I[MAX_SYMBOLS],
    fixed_point_t output_symbols_Q[MAX_SYMBOLS]
) {
    // QPSK constellation points
    const fixed_point_t qpsk_i[4] = {0.7071f, -0.7071f, 0.7071f, -0.7071f};
    const fixed_point_t qpsk_q[4] = {0.7071f, 0.7071f, -0.7071f, -0.7071f};

    int bits_per_symbol = 2; // QPSK uses 2 bits per symbol
    int max_symbols_to_process = num_bits / bits_per_symbol;
    if (max_symbols_to_process > MAX_SYMBOLS) {
        max_symbols_to_process = MAX_SYMBOLS;
    }

    for (int i = 0; i < max_symbols_to_process; ++i) {
        int bit_pos = i * bits_per_symbol;
        int byte_idx = bit_pos / 8;
        int bit_offset_in_byte = bit_pos % 8;

        // Check for array bounds before access
        if (byte_idx >= MAX_INPUT_BYTES) {
            output_symbols_I[i] = 0;
            output_symbols_Q[i] = 0;
            continue;
        }

        uint8_t symbol_bits;
        if (bit_offset_in_byte <= 6) { // If bits are fully within the current byte
            symbol_bits = ((uint8_t)input_bytes[byte_idx] >> bit_offset_in_byte) & 0x3;
        } else { // Bits are split across two bytes
            if (byte_idx + 1 >= MAX_INPUT_BYTES) { // Check for next byte boundary
                output_symbols_I[i] = 0;
                output_symbols_Q[i] = 0;
                continue;
            }
            // Extract bits from current byte and next byte
            uint8_t byte0 = (uint8_t)input_bytes[byte_idx];
            uint8_t byte1 = (uint8_t)input_bytes[byte_idx + 1];

            // Combine bits. Example: if bit_offset_in_byte is 7, 1 bit from byte0, 1 from byte1
            // (byte0 >> 7) & 0x1  (LSB of the two-bit symbol from current byte)
            // (byte1 << 1) & 0x2  (MSB of the two-bit symbol from next byte)
            // Assuming little-endian bit packing within symbol:
            symbol_bits = ((byte0 >> bit_offset_in_byte) | (byte1 << (8 - bit_offset_in_byte))) & 0x3;
        }

        // Map to constellation points
        output_symbols_I[i] = qpsk_i[symbol_bits];
        output_symbols_Q[i] = qpsk_q[symbol_bits];
    }
}
