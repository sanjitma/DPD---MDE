```markdown
# DPD-MDE: Digital Pre-Distortion with Modified Differential Evolution (CUDA Accelerated)

## Overview

This project implements a Digital Pre-Distortion (DPD) system designed to linearize the non-linear behavior of Power Amplifiers (PAs) in communication systems. It utilizes a novel approach based on a Modified Differential Evolution (MDE) algorithm for adaptive DPD coefficient estimation. A significant aspect of this implementation is its acceleration using NVIDIA CUDA, offloading computationally intensive signal processing blocks to the GPU for improved performance.

The system simulates a complete digital communication transmit and receive chain, including constellation mapping, pulse shaping, digital-to-analog conversion (DAC), quadrature modulation (QM), digital up-conversion (DUC), a Saleh Power Amplifier (PA) model, digital down-conversion (DDC), analog-to-digital conversion (ADC), and a feedback pulse shaping filter.

## Motivation

Power Amplifiers (PAs) are critical components in wireless communication systems, but they inherently introduce non-linear distortions, especially when operating near saturation for efficiency. These non-linearities cause spectral regrowth (out-of-band emissions) and in-band distortion, degrading signal quality and interfering with adjacent channels. Digital Pre-Distortion (DPD) is a widely adopted technique to compensate for these non-linearities by applying an inverse distortion characteristic to the signal *before* it enters the PA.

This project explores the use of a Modified Differential Evolution (MDE) algorithm for DPD adaptation, offering a robust optimization method to find the optimal pre-distortion coefficients, and leverages CUDA to accelerate the demanding signal processing tasks.

## System Architecture

The DPD system operates in three distinct modes: `BASELINE`, `ADAPT`, and `FINAL`. The overall signal flow is as follows:

**Operating Modes:**

* **`BASELINE` Mode**: The DPD block acts as a passthrough (no pre-distortion applied). This mode captures the raw non-linear behavior of the PA.
* **`ADAPT` Mode**: The MDE algorithm continuously updates the DPD coefficients (`w`) based on the comparison between the desired input signal and the feedback signal from the PA output. This process typically runs for multiple iterations to converge on optimal coefficients.
* **`FINAL` Mode**: The DPD block applies the learned (adapted) coefficients to the input signal, demonstrating the linearization effect on the PA output.

## Key Features

* **Digital Pre-Distortion (DPD)**: Implements a memory polynomial model for DPD.
* **Modified Differential Evolution (MDE)**: Utilizes an evolutionary algorithm for robust and adaptive coefficient estimation.
* **Comprehensive Signal Chain**: Includes essential digital signal processing (DSP) blocks:
    * Constellation Mapper (QPSK)
    * Raised Cosine Pulse Shaping Filter
    * Multi-bit Digital-to-Analog Converter (DAC)
    * Digital Quadrature Modulator (QM) with NCO
    * Digital Up-Converter (DUC)
    * Saleh Power Amplifier (PA) Non-linear Model
    * Digital Down-Converter (DDC) with FIR filter and NCO
    * Dual Analog-to-Digital Converter (ADC)
    * Feedback Pulse Shaping Filter
* **Indirect Learning Architecture**: The DPD adaptation is performed in a feedback loop, comparing the pre-distorted signal (after PA) with the original desired signal.
* **CUDA Acceleration**: Key computational blocks are implemented as CUDA kernels (`__global__` functions) to leverage GPU parallelism.
* **Detailed Output**: Generates `.txt` files at various stages of the signal chain for analysis and visualization of the signal processing effects and DPD performance.

## CUDA Implementation Details

The project has been refactored to be CUDA compatible, enabling significant portions of the signal processing to run on NVIDIA GPUs.

* **CUDA Kernels (`__global__`)**: Functions like `ddc_demodulator_kernel`, `digital_upconverter_kernel`, `final_dpd_dac_qm_kernel`, `amplifier_kernel`, and `normalize_kernel` are implemented as CUDA kernels. These kernels are launched from host (CPU) code and execute in parallel on the GPU.
* **Device Functions (`__device__`)**: Helper functions and core DSP algorithms (e.g., `legendre_poly`, `compute_phi_all`, `generate_random_float`, `generate_random_index`, `compute_fitness`, `dac_multibit_with_select`, `nco`, `digital_qm`, `raised_cosine_filter`, `convolve`, `fir_filter`, `saleh_amplifier`) are marked `__device__` or `__host__ __device__`, allowing them to be called from within CUDA kernels.
* **Constant Memory (`__constant__`)**: Look-up tables (LUTs) for NCO (sine/cosine) and FIR filter coefficients are stored in `__constant__` memory on the device. This read-only memory is optimized for fast access by all threads in a kernel.
* **Memory Management**: Explicit CUDA API calls (`cudaMalloc`, `cudaMemcpy`, `cudaFree`) are used to allocate and deallocate memory on the GPU, and to transfer data efficiently between the host and device.
* **DPD Adaptation (MDE Algorithm)**: The core Differential Evolution (DE) algorithm (`dpd_mde`) itself, which manages the population and evolves the DPD coefficients, remains a host-side function due to its complex state management and iterative nature. However, the *application* of the DPD memory polynomial (calculating the pre-distorted signal using the current weights) is parallelized on the GPU within the `final_dpd_dac_qm_kernel`.
* **Type Replacement**: HLS-specific fixed-point and arbitrary-precision integer types (e.g., `ap_fixed`, `ap_uint`) have been replaced with standard C++ `float` and `uint32_t`/`int16_t` types, suitable for CUDA compilation.

## Getting Started

To build and run this project, you will need a system with an NVIDIA GPU and the CUDA Toolkit installed.

### Prerequisites

* **NVIDIA GPU**: A CUDA-enabled graphics card.
* **CUDA Toolkit**: Version 10.2 or higher (including `nvcc` compiler).
* **C++ Compiler**: A C++ compiler compatible with your CUDA Toolkit (e.g., GCC).
* **`i_symbols.txt` and `q_symbols.txt`**: These input files are expected to be present in the same directory as your executable or at the specified path in `test_for_mde.cpp`. (The provided `test_for_mde.cpp` expects them at `C:/Users/SKrss/Modified_DE_2/Modified_DE/`).

### Build Instructions

Navigate to the root directory of the project in your terminal. You can compile the project using `nvcc`:

```bash
nvcc -o dpd_mde_sim \
     test_for_mde.cpp \
     adc_mdejoin.cu \
     circuit_mdefinal.cu \
     conste_mdejoin.cu \
     dac_mdejoin.cu \
     ddc_mdejoin.cu \
     duc_mdejoin.cu \
     mde-mai.cu \
     pa_mdejoin.cu \
     psf_mdejoin.cu \
     qm_mdejoin.cu \
     -I. -lcudart -lm
````

  * `-o dpd_mde_sim`: Specifies the output executable name.
  * `*.cu`: All CUDA source files.
  * `*.cpp`: All C++ source files (if any remain that don't need `.cu`).
  * `-I.`: Adds the current directory to the include path for header files.
  * `-lcudart`: Links against the CUDA Runtime library.
  * `-lm`: Links against the math library (for `sqrt`, `fabs`, etc.).

**Note**: Ensure that the paths for `i_symbols.txt` and `q_symbols.txt` in `test_for_mde.cpp` match the actual location of these files on your system, or place them in the same directory as the compiled executable.



The program will execute the DPD simulation in three phases:

1.  **Baseline Run**: Simulates the system without DPD.
2.  **Adaptation Phase**: Runs the MDE algorithm for 75 iterations to learn DPD coefficients.
3.  **Final Run**: Simulates the system with the learned DPD coefficients applied.

During execution, it will print a "Done. All outputs written to files." message upon completion.

## Output Files

The simulation generates several `.txt` files in the current working directory (or the directory from which the executable is run). These files contain floating-point values representing the I/Q components or other metrics at different stages of the signal chain, which can be used for plotting and analysis (e.g., in MATLAB, Python with Matplotlib, or Octave).

  * `output_conste_i.txt`, `output_conste_q.txt`: I and Q components after constellation mapping.
  * `output_psf_i.txt`, `output_psf_q.txt`: I and Q components after pulse shaping (feedforward path).
  * `output_dpd_i.txt`, `output_dpd_q.txt`: I and Q components after DPD application.
  * `output_dac_i.txt`, `output_dac_q.txt`: I and Q components after DAC.
  * `output_qm.txt`: Output of the Quadrature Modulator.
  * `output_duc.txt`: Output of the Digital Up-Converter.
  * `output_amp_i.txt`, `output_amp_q.txt`: I and Q components after the Saleh Power Amplifier.
  * `output_amp_magnitude.txt`: Magnitude of the signal after the Power Amplifier.
  * `output_amp_gain_lin.txt`, `output_amp_gain_db.txt`: Linear and dB gain of the Power Amplifier.
  * `output_ddc_i.txt`, `output_ddc_q.txt`: I and Q components after Digital Down-Converter.
  * `output_adc_i.txt`, `output_adc_q.txt`: I and Q components after Analog-to-Digital Converter.
  * `output_i_pa_psf_fb_no_dpd.txt`, `output_q_pa_psf_fb_no_dpd.txt`: I and Q components of the feedback pulse-shaped signal during the `BASELINE` run (before DPD adaptation).
  * `output_psf_fb_i_with_dpd.txt`, `output_psf_fb_q_with_dpd.txt`: I and Q components of the feedback pulse-shaped signal during the `FINAL` run (after DPD adaptation).

## Contributing

Contributions are welcome\! If you find bugs, have suggestions for improvements, or want to add new features, please feel free to open an issue or submit a pull request.

## Acknowledgements

  * **Digital Pre-Distortion (DPD)**: Based on common DPD principles and memory polynomial models.
  * **Modified Differential Evolution (MDE)**: Inspired by Differential Evolution optimization algorithms.
  * **Saleh Power Amplifier Model**: A widely used behavioral model for non-linear amplifiers.
  * **NVIDIA CUDA**: For enabling high-performance parallel computation.

<!-- end list -->

```
```
