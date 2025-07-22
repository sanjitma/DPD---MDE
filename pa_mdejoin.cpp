#include "pa_mdejoiner.h"
#include <cmath>
#include <iostream>

void saleh_amplifier(
    data_t in_i,
    data_t in_q,
    data_t& out_i,
    data_t& out_q,
    data_t& magnitude,
    data_t& gain_lin,
    data_t& gain_db
) {
    // Saleh model parameters
    const float alpha_a = 4.0;  // Amplitude coefficient
    const float beta_a = 0.7;   // Amplitude coefficient
    const float alpha_p = 0.15; // Phase coefficient
    const float beta_p = 0.25;  // Phase coefficient

    // Calculate input magnitude
    float i_float = in_i;
    float q_float = in_q;
    float r = std::sqrt(i_float*i_float + q_float*q_float);

    // Calculate input phase
    float phi = std::atan2(q_float, i_float);

    // Saleh model for amplitude
    float A_r = (alpha_a * r) / (1 + beta_a * r * r);

    // Saleh model for phase
    float P_r = (alpha_p * r * r) / (1 + beta_p * r * r);

    // Convert back to I/Q
    float out_i_float = A_r * std::cos(phi + P_r);
    float out_q_float = A_r * std::sin(phi + P_r);

    // FIX: Multiply by 3.4 to scale from ±2.36 to ±8 range
    out_i_float *= 5.0;
    out_q_float *= 5.0;

    // Set output
    out_i = data_t(out_i_float);
    out_q = data_t(out_q_float);
    magnitude = data_t(A_r);

    // Calculate gain
    if (r > 0.001f) {
        gain_lin = data_t(A_r / r);
        gain_db = data_t(20 * std::log10(A_r / r));
    } else {
        gain_lin = data_t(1.0);
        gain_db = data_t(0.0);
    }
}
