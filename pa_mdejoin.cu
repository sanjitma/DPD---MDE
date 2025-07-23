// File: C:\Modified_DE_cuda\pa_mdejoin.cu

#include "pa_mdejoiner.h"
#include <cmath> // For sqrtf, sinf, cosf, log10f

// This is the correct implementation for the Saleh Power Amplifier model.
// It should be in this file and this file only.
__host__ __device__ void saleh_amplifier(
    data_t in_i,
    data_t in_q,
    data_t& out_i,
    data_t& out_q,
    data_t& magnitude,
    data_t& gain_lin,
    data_t& gain_db
) {
    // --- Saleh Model Parameters (typical values) ---
    const float alpha_a = 2.1587f;  // AM/AM gain
    const float beta_a  = 1.1517f;  // AM/AM compression
    const float alpha_p = 4.0033f;  // AM/PM shift
    const float beta_p  = 9.1040f;  // AM/PM shift

    // Epsilon to avoid division by zero
    const float epsilon = 1e-9f;

    // 1. Calculate input magnitude squared and magnitude
    data_t r_sq = in_i * in_i + in_q * in_q;
    data_t r = sqrtf(r_sq);

    // 2. Calculate AM/AM distortion (output magnitude)
    data_t A_r = (alpha_a * r) / (1.0f + beta_a * r_sq);

    // 3. Calculate AM/PM distortion (phase shift in radians)
    data_t Phi_r = (alpha_p * r_sq) / (1.0f + beta_p * r_sq);

    // 4. Apply gain and phase shift
    data_t gain = (r > epsilon) ? (A_r / r) : alpha_a;

    // Apply the gain to the input samples
    data_t g_in_i = gain * in_i;
    data_t g_in_q = gain * in_q;

    // Apply the phase rotation
    float cos_phi = cosf(Phi_r);
    float sin_phi = sinf(Phi_r);
    out_i = g_in_i * cos_phi - g_in_q * sin_phi;
    out_q = g_in_i * sin_phi + g_in_q * cos_phi;

    // 5. Calculate output metrics for analysis
    magnitude = A_r;
    gain_lin = gain;
    gain_db = 20.0f * log10f(gain_lin > epsilon ? gain_lin : epsilon);
}
