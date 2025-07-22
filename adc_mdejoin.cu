#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define N 8192
#define W 16  // Output bit width


// CUDA-compatible types
typedef float adc_in_t;
typedef int16_t adc_out_t;

// CUDA kernel for dual ADC conversion
__global__ void dual_adc_kernel(const adc_in_t *I_analog_in, const adc_in_t *Q_analog_in, adc_out_t *I_digital_out, adc_out_t *Q_digital_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float V_REF = 1.0f;
        const float V_MIN = -V_REF;
        const float V_MAX = V_REF;
        const int scale = (1 << (W-1)) - 1; // 32767 for 16-bit

        // I channel
        float clamped_I = (I_analog_in[i] > V_MAX) ? V_MAX : ((I_analog_in[i] < V_MIN) ? V_MIN : I_analog_in[i]);
        int16_t scaled_I = (int16_t)(clamped_I * scale);
        I_digital_out[i] = scaled_I;

        // Q channel
        float clamped_Q = (Q_analog_in[i] > V_MAX) ? V_MAX : ((Q_analog_in[i] < V_MIN) ? V_MIN : Q_analog_in[i]);
        int16_t scaled_Q = (int16_t)(clamped_Q * scale);
        Q_digital_out[i] = scaled_Q;
    }
}

// Host wrapper for dual ADC system
void dual_adc_system(
    const adc_in_t I_analog_in[N], const adc_in_t Q_analog_in[N],
    adc_out_t I_digital_out[N], adc_out_t Q_digital_out[N]
) {
    adc_in_t *d_I_analog_in, *d_Q_analog_in;
    adc_out_t *d_I_digital_out, *d_Q_digital_out;
    cudaMalloc(&d_I_analog_in, N * sizeof(adc_in_t));
    cudaMalloc(&d_Q_analog_in, N * sizeof(adc_in_t));
    cudaMalloc(&d_I_digital_out, N * sizeof(adc_out_t));
    cudaMalloc(&d_Q_digital_out, N * sizeof(adc_out_t));
    cudaMemcpy(d_I_analog_in, I_analog_in, N * sizeof(adc_in_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q_analog_in, Q_analog_in, N * sizeof(adc_in_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dual_adc_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_I_analog_in, d_Q_analog_in, d_I_digital_out, d_Q_digital_out);
    cudaDeviceSynchronize();

    cudaMemcpy(I_digital_out, d_I_digital_out, N * sizeof(adc_out_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(Q_digital_out, d_Q_digital_out, N * sizeof(adc_out_t), cudaMemcpyDeviceToHost);

    cudaFree(d_I_analog_in);
    cudaFree(d_Q_analog_in);
    cudaFree(d_I_digital_out);
    cudaFree(d_Q_digital_out);

    // Debug output
    for (int i = 0; i < 10; ++i) {
        printf("Sample %d: I_analog=%f, I_digital=%d | Q_analog=%f, Q_digital=%d\n",
               i, I_analog_in[i], I_digital_out[i], Q_analog_in[i], Q_digital_out[i]);
    }
}
