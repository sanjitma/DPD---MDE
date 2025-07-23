// File: C:\Modified_DE_cuda\pa_mdejoin.cu
#include "pa_mdejoiner.h"
#include <cmath>

// FIXED: This is the correct code for the Saleh Power Amplifier
__host__ __device__ void saleh_amplifier(
    data_t in_i, data_t in_q,
    data_t& out_i, data_t& out_q,
    data_t& magnitude, data_t& gain_lin, data_t& gain_db
) {
    const float alpha_a = 2.1587f, beta_a  = 1.1517f;
    const float alpha_p = 4.0033f, beta_p  = 9.1040f;
    const float epsilon = 1e-9f;

    data_t r_sq = in_i * in_i + in_q * in_q;
    data_t r = sqrtf(r_sq);
    data_t A_r = (alpha_a * r) / (1.0f + beta_a * r_sq);
    data_t Phi_r = (alpha_p * r_sq) / (1.0f + beta_p * r_sq);
    data_t gain = (r > epsilon) ? (A_r / r) : alpha_a;
    data_t g_in_i = gain * in_i;
    data_t g_in_q = gain * in_q;
    float cos_phi = cosf(Phi_r);
    float sin_phi = sinf(Phi_r);
    out_i = g_in_i * cos_phi - g_in_q * sin_phi;
    out_q = g_in_i * sin_phi + g_in_q * cos_phi;
    magnitude = A_r;
    gain_lin = gain;
    gain_db = 20.0f * log10f(gain_lin > epsilon ? gain_lin : epsilon);
}
