# FastXC
High Performance Noise Cross-Corelation Computing Code for 9-component Recordings

This program is designed to efficiently compute single/nine-component noise cross-correlation functions (NCFs) from ambient noise data using a high-performance CPU-GPU heterogeneous computing framework. It integrates data preprocessing, accelerated cross-correlation computation, and various stacking techniques (Linear, PWS, tf-PWS), with a particular emphasis on optimizing the computing process using CUDA technology. This significantly enhances processing speed and the signal-to-noise ratio of the data, making it especially suitable for handling large-scale noise datasets.

## Quick Start
cd FastXC


make veryclean


make


python run.py


## Environment Setup

To run this CUDA program, ensure your system meets the following requirements:

1. **NVIDIA GPU**: Your computer must be equipped with an NVIDIA GPU that supports CUDA.
2. **CUDA Toolkit**: You must install the CUDA Toolkit, essential for running CUDA programs. You can download the latest version of the CUDA Toolkit from the [NVIDIA official website](https://developer.nvidia.com/cuda-downloads).
3. **GPU Drivers**: Make sure that your NVIDIA GPU drivers are up-to-date to be compatible with the installed version of CUDA.

### Tools Check

Before starting, it is recommended to use the following commands to check if your environment is correctly configured:

- Use the `nvidia-smi` command to check the status of your GPU and drivers.
```bash
  nvidia-smi
```
This command will display details about your GPU and the current version of drivers.
```bash
  nvcc --version
```
This command help to confirm the version of CUDA and CUDA compiler (NVCC).
