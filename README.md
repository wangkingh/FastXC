* Switch Language 切换语言: [English](README.md)[英文], [简体中文](README.zh-CN.md)*[Simplified Chinese]
# FastXC
High Performance Noise Cross-Corelation Computing Code for 9-component Recordings

Using a high-performance CPU-GPU heterogeneous computing framework, this program is designed to efficiently compute single/nine-component noise cross-correlation functions (NCFs) from ambient noise data. It integrates data preprocessing, accelerated cross-correlation computation, and various stacking techniques (Linear, PWS, tf-PWS), particularly optimizing the computing process using CUDA technology. This significantly enhances processing speed and the signal-to-noise ratio of the data, making it especially suitable for handling large-scale noise datasets.

## Installation
If you are NEW to using a CUDA-based program, Please check the [Environment Setup](#environment-setup) first.
### Python Version
- **Required**: Python 3.8 or higher version.

### Required Third-party Python Modules
- **Libraries**:`obspy`, `pandas`, `scipy`,`matplotlib`, `tqdm`, `numpy`.
- Install the latest versions of these libraries, as only their basic functionalities are utilized. Use the following command to install or confirm installation:
```bash
pip install obspy pandas scipy matplotlib tqdm numpy
```
Or you can use anaconda if you are familiar with that (very suggested choice).

### Compilation
The code is separated into two parts. The `Python` part is designed to allocate computing tasks and address other higher-level issues, while the fundamental computing part is finished using `CUDA-C` or `C.`
For those written in `CUDA-C` or `C`, a compilation step is required.



## Quick Start
cd FastXC


make veryclean


make


python run.py

## Start to do XC (Cross-Correlation)
## Editing the configure file
## Advanced Options

## Environment Setup

To run this CUDA program, ensure your system meets the following requirements:

1. **NVIDIA GPU**: Your computer must have an NVIDIA GPU that supports CUDA.
2. **CUDA Toolkit**: You must install the CUDA Toolkit, which is essential for running CUDA programs. You can download the latest version from the [NVIDIA official website](https://developer.nvidia.com/cuda-downloads).
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
This command helps confirm the CUDA and CUDA compiler (NVCC) version.

## Changelog
see [Changelog](changelog.md)
## Contact

If you have any questions or suggestions or want to contribute to the project, open an [issue](https://github.com/wangkingh/FastXC/issues) or submit a pull request.

For more direct inquiries, you can reach the author at:  
**Email:** [wkh16@mail.ustc.edu.cn](mailto:wkh16@mail.ustc.edu.cn)
