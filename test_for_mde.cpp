
#include <iostream>
#include <fstream>
#include <stdint.h>
#include "duc_mdejoiner.h"
#include "pa_mdejoiner.h"
#include "ddc_mdejoiner.h"
#include "mde.h"

#define MAX_INPUT_BYTES 8192
#define DATA_LEN 8192
#define MAX_SYMBOLS 32768
#define INTERPOLATION_FACTOR 8
#define DECIM_FACTOR 8


typedef float fixed_t;
typedef float data_t;
typedef float sample_type;
typedef float baseband_t;
typedef float adc_out_t;
typedef float data_ty;

enum DPDMode { BASELINE, ADAPT, FINAL };

void circuit_final(
    float input_bytes[MAX_INPUT_BYTES],
    int num_bits,
    sample_type duc_out[DATA_LEN * INTERPOLATION_FACTOR],
    fixed_t i_symbols[MAX_SYMBOLS],
    fixed_t q_symbols[MAX_SYMBOLS],
    fixed_t i_psf[DATA_LEN],
    fixed_t q_psf[DATA_LEN],
    data_t dpd_i[DATA_LEN],
    data_t dpd_q[DATA_LEN],
    data_ty dac_i_arr[DATA_LEN],
    data_ty dac_q_arr[DATA_LEN],
    data_t qm_out_buf[DATA_LEN],
    data_t amp_out_i[DATA_LEN * INTERPOLATION_FACTOR],
    data_t amp_out_q[DATA_LEN * INTERPOLATION_FACTOR],
    data_t amp_magnitude[DATA_LEN * INTERPOLATION_FACTOR],
    data_t amp_gain_lin[DATA_LEN * INTERPOLATION_FACTOR],
    data_t amp_gain_db[DATA_LEN * INTERPOLATION_FACTOR],
    baseband_t ddc_i_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
    baseband_t ddc_q_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
    adc_out_t adc_i_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
    adc_out_t adc_q_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
    fixed_t i_psf_fb[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
    fixed_t q_psf_fb[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
    DPDMode mode
);

int main()
{
    float input_bytes[MAX_INPUT_BYTES] = {0};
    int num_bits = 0;

    static sample_type duc_out[DATA_LEN * INTERPOLATION_FACTOR] = {0};
    static fixed_t i_symbols[MAX_SYMBOLS] = {0};
    static fixed_t q_symbols[MAX_SYMBOLS] = {0};
    static fixed_t i_psf[DATA_LEN] = {0};
    static fixed_t q_psf[DATA_LEN] = {0};
    static data_t dpd_i[DATA_LEN] = {0};
    static data_t dpd_q[DATA_LEN] = {0};
    static data_ty dac_i_arr[DATA_LEN] = {0};
    static data_ty dac_q_arr[DATA_LEN] = {0};
    static data_t qm_out_buf[DATA_LEN] = {0};

    static data_t amp_out_i[DATA_LEN * INTERPOLATION_FACTOR] = {0};
    static data_t amp_out_q[DATA_LEN * INTERPOLATION_FACTOR] = {0};
    static data_t amp_magnitude[DATA_LEN * INTERPOLATION_FACTOR] = {0};
    static data_t amp_gain_lin[DATA_LEN * INTERPOLATION_FACTOR] = {0};
    static data_t amp_gain_db[DATA_LEN * INTERPOLATION_FACTOR] = {0};

    static baseband_t ddc_i_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR] = {0};
    static baseband_t ddc_q_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR] = {0};

    static adc_out_t adc_i_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR] = {0};
    static adc_out_t adc_q_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR] = {0};

    static fixed_t i_psf_fb[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR] = {0};
    static fixed_t q_psf_fb[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR] = {0};

    std::ifstream i_file("C:/Users/SKrss/Modified_DE_2/Modified_DE/i_symbols.txt");
    std::ifstream q_file("C:/Users/SKrss/Modified_DE_2/Modified_DE/q_symbols.txt");
    if (!i_file || !q_file) {
        std::cerr << "Cannot open i_symbols.txt or q_symbols.txt\n";
        return 1;
    }
    double val;
    for (int i = 0; i < DATA_LEN; ++i) {
        if (!(i_file >> val)) { std::cerr << "Not enough data in i_symbols.txt\n"; return 1; }
        i_symbols[i] = fixed_t(val) * fixed_t(2.5);
        if (!(q_file >> val)) { std::cerr << "Not enough data in q_symbols.txt\n"; return 1; }
        q_symbols[i] = fixed_t(val) * fixed_t(2.5);
    }
    i_file.close();
    q_file.close();

    // 1. Run baseline (no DPD adaptation, DPD is passthrough)
    circuit_final(input_bytes, num_bits, duc_out,
                  i_symbols, q_symbols, i_psf, q_psf,
                  dpd_i, dpd_q, dac_i_arr, dac_q_arr, qm_out_buf,
                  amp_out_i, amp_out_q, amp_magnitude, amp_gain_lin, amp_gain_db,
                  ddc_i_out, ddc_q_out,
                  adc_i_out, adc_q_out,
                  i_psf_fb, q_psf_fb,
                  BASELINE);

    int adc_len = (DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR;
    std::ofstream pa_out_file_i("output_i_pa_psf_fb_no_dpd.txt");
    std::ofstream pa_out_file_q("output_q_pa_psf_fb_no_dpd.txt");
    for (int i = 0; i < adc_len; ++i) {
        pa_out_file_i << static_cast<double>(i_psf_fb[i]) << "\n";
        pa_out_file_q << static_cast<double>(q_psf_fb[i]) << "\n";
    }
    pa_out_file_i.close();
    pa_out_file_q.close();

    // 2. Run adaptation (DPD learns)
    for (int adapt_itr = 0; adapt_itr < 75; ++adapt_itr) {
        circuit_final(input_bytes, num_bits, duc_out,
                      i_symbols, q_symbols, i_psf, q_psf,
                      dpd_i, dpd_q, dac_i_arr, dac_q_arr, qm_out_buf,
                      amp_out_i, amp_out_q, amp_magnitude, amp_gain_lin, amp_gain_db,
                      ddc_i_out, ddc_q_out,
                      adc_i_out, adc_q_out,
                      i_psf_fb, q_psf_fb,
                      ADAPT);
    }

    // 3. Run transmit chain again with learned DPD (no further adaptation)
    circuit_final(input_bytes, num_bits, duc_out,
                  i_symbols, q_symbols, i_psf, q_psf,
                  dpd_i, dpd_q, dac_i_arr, dac_q_arr, qm_out_buf,
                  amp_out_i, amp_out_q, amp_magnitude, amp_gain_lin, amp_gain_db,
                  ddc_i_out, ddc_q_out,
                  adc_i_out, adc_q_out,
                  i_psf_fb, q_psf_fb,
                  FINAL);

    std::ofstream psf_fb_i_file("output_psf_fb_i_with_dpd.txt");
    std::ofstream psf_fb_q_file("output_psf_fb_q_with_dpd.txt");
    for (int i = 0; i < adc_len; ++i) {
        psf_fb_i_file << static_cast<double>(i_psf_fb[i]) << "\n";
        psf_fb_q_file << static_cast<double>(q_psf_fb[i]) << "\n";
    }
    psf_fb_i_file.close();
    psf_fb_q_file.close();

    // Write outputs to files for each stage (unchanged)
    std::ofstream f_conste_i("output_conste_i.txt");
    std::ofstream f_conste_q("output_conste_q.txt");
    for (int i = 0; i < MAX_SYMBOLS; ++i) {
        f_conste_i << static_cast<double>(i_symbols[i]) << "\n";
        f_conste_q << static_cast<double>(q_symbols[i]) << "\n";
    }
    f_conste_i.close();
    f_conste_q.close();

    std::ofstream f_psf_i("output_psf_i.txt");
    std::ofstream f_psf_q("output_psf_q.txt");
    for (int i = 0; i < DATA_LEN; ++i) {
        f_psf_i << static_cast<double>(i_psf[i]) << "\n";
        f_psf_q << static_cast<double>(q_psf[i]) << "\n";
    }
    f_psf_i.close();
    f_psf_q.close();

    std::ofstream f_dpd_i("output_dpd_i.txt");
    std::ofstream f_dpd_q("output_dpd_q.txt");
    for (int i = 0; i < DATA_LEN; ++i) {
        f_dpd_i << static_cast<double>(dpd_i[i]) << "\n";
        f_dpd_q << static_cast<double>(dpd_q[i]) << "\n";
    }
    f_dpd_i.close();
    f_dpd_q.close();

    std::ofstream f_dac_i("output_dac_i.txt");
    std::ofstream f_dac_q("output_dac_q.txt");
    for (int i = 0; i < DATA_LEN; ++i) {
        f_dac_i << static_cast<double>(dac_i_arr[i]) << "\n";
        f_dac_q << static_cast<double>(dac_q_arr[i]) << "\n";
    }
    f_dac_i.close();
    f_dac_q.close();

    std::ofstream f_qm("output_qm.txt");
    for (int i = 0; i < DATA_LEN; ++i) {
        f_qm << static_cast<double>(qm_out_buf[i]) << "\n";
    }
    f_qm.close();

    std::ofstream duc_outfile("output_duc.txt");
    for (int i = 0; i < DATA_LEN * INTERPOLATION_FACTOR; ++i) {
        duc_outfile << static_cast<double>(duc_out[i]) << "\n";
    }
    duc_outfile.close();

    std::ofstream amp_i_file("output_amp_i.txt");
    std::ofstream amp_q_file("output_amp_q.txt");
    std::ofstream amp_mag_file("output_amp_magnitude.txt");
    std::ofstream amp_gain_lin_file("output_amp_gain_lin.txt");
    std::ofstream amp_gain_db_file("output_amp_gain_db.txt");
    for (int i = 0; i < DATA_LEN * INTERPOLATION_FACTOR; ++i) {
        amp_i_file << static_cast<double>(amp_out_i[i]) << "\n";
        amp_q_file << static_cast<double>(amp_out_q[i]) << "\n";
        amp_mag_file << static_cast<double>(amp_magnitude[i]) << "\n";
        amp_gain_lin_file << static_cast<double>(amp_gain_lin[i]) << "\n";
        amp_gain_db_file << static_cast<double>(amp_gain_db[i]) << "\n";
    }
    amp_i_file.close();
    amp_q_file.close();
    amp_mag_file.close();
    amp_gain_lin_file.close();
    amp_gain_db_file.close();

    std::ofstream ddc_i_file("output_ddc_i.txt");
    std::ofstream ddc_q_file("output_ddc_q.txt");
    int ddc_len = (DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR;
    for (int i = 0; i < ddc_len; ++i) {
        ddc_i_file << static_cast<double>(ddc_i_out[i]) << "\n";
        ddc_q_file << static_cast<double>(ddc_q_out[i]) << "\n";
    }
    ddc_i_file.close();
    ddc_q_file.close();

    std::ofstream adc_i_file("output_adc_i.txt");
    std::ofstream adc_q_file("output_adc_q.txt");
    for (int i = 0; i < adc_len; ++i) {
        adc_i_file << adc_i_out[i] << "\n";
        adc_q_file << adc_q_out[i] << "\n";
    }
    adc_i_file.close();
    adc_q_file.close();

    std::cout << "Done. All outputs written to files.\n";
    return 0;
}
