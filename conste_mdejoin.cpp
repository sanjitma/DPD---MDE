// CUDA port: Remove HLS/Vivado includes
#include <stdint.h>
#include <cmath>

typedef float fixed_point_t;
#define MAX_INPUT_BYTES 8192
#define MAX_SYMBOLS 32768
typedef int modulation_type;

void conste(
    const float input_bytes[MAX_INPUT_BYTES],
    int num_bits,
    fixed_point_t output_symbols_I[MAX_SYMBOLS],
    fixed_point_t output_symbols_Q[MAX_SYMBOLS]
) {
    // QPSK constellation points
    const fixed_point_t qpsk_i[4] = {0.7071f, -0.7071f, 0.7071f, -0.7071f};
    const fixed_point_t qpsk_q[4] = {0.7071f, 0.7071f, -0.7071f, -0.7071f};

    int bits_per_symbol = 2; // QPSK uses 2 bits per symbol
    int max_symbols = num_bits / bits_per_symbol;
    if (max_symbols > MAX_SYMBOLS) {
        max_symbols = MAX_SYMBOLS;
    }

    for (int i = 0; i < max_symbols; i++) {
        int bit_pos = i * bits_per_symbol;
        int byte_idx = bit_pos / 8;
        int bit_offset = bit_pos % 8;

        if (byte_idx >= MAX_INPUT_BYTES || byte_idx + 1 >= MAX_INPUT_BYTES) {
            output_symbols_I[i] = 0;
            output_symbols_Q[i] = 0;
            continue;
        }

        // Extract the symbol bits
        uint8_t byte0 = (uint8_t)input_bytes[byte_idx];
        uint8_t byte1 = (uint8_t)input_bytes[byte_idx + 1];
        uint8_t symbol_bits;
        if (bit_offset <= 6) {
            symbol_bits = (byte0 >> bit_offset) & 0x3;
        } else {
            symbol_bits = ((byte0 >> bit_offset) | (byte1 << (8 - bit_offset))) & 0x3;
        }
        output_symbols_I[i] = qpsk_i[symbol_bits];
        output_symbols_Q[i] = qpsk_q[symbol_bits];
    }
}
            symbol_bits = (input_bytes[byte_idx] >> (6 - bit_offset)) & 0x3;
        } else {
            // Bits are split across two bytes
            symbol_bits = ((input_bytes[byte_idx] & 0x1) << 1) |
                          ((input_bytes[byte_idx + 1] >> 7) & 0x1);
        }

        // Map to constellation points
        output_symbols_I[i] = qpsk_i[symbol_bits];
        output_symbols_Q[i] = qpsk_q[symbol_bits];
    }
}
