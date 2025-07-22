#ifndef ADC_HERE_F_H
#define ADC_HERE_F_H

#include <stdint.h>

#define N 8192
#define W 16  // Output bit width

typedef float adc_in_t;
typedef int16_t adc_out_t;

void dual_adc_system(
    const adc_in_t I_analog_in[N],
    const adc_in_t Q_analog_in[N],
    adc_out_t I_digital_out[N],
    adc_out_t Q_digital_out[N]
);

#endif // ADC_HERE_F_H
