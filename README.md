<p align="center">
  <img src="https://private-user-images.githubusercontent.com/38589094/394993541-1550eb2f-23be-4795-a48a-7f1a12144604.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzM5NzM1MTksIm5iZiI6MTczMzk3MzIxOSwicGF0aCI6Ii8zODU4OTA5NC8zOTQ5OTM1NDEtMTU1MGViMmYtMjNiZS00Nzk1LWE0OGEtN2YxYTEyMTQ0NjA0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMTIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjEyVDAzMTMzOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZiZWU3OTQyMDUxYzVmYzgzMGFmY2QwMTI2MTAxZjJhODkxN2RiYzljNzFlNGNkYzRlNzAxMzY3ZmI4ZTAzMmUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.t8WLt2ABoxLwoFi4lLLLOZ1f-c5JYcklrYRsKX4fVbk" alt="广告图片" width="300">
</p>

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
For those written in `CUDA-C` or `C`, a compilation step is required. Following the following steps:

```bash
cd FastXC
make veryclean
make
```


## Quick Start
```python
python run.py
```

## Start to do XC (Cross-Correlation)
## Editing the configure file
## Advanced Options

## Environment Setup

To run this CUDA program, ensure your system meets the following requirements:

1. **NVIDIA GPU**: Your computer must have an NVIDIA GPU that supports CUDA.
2. **CUDA Toolkit**: You must install the CUDA Toolkit, which is essential for running CUDA programs. The latest version can be downloaded from the [NVIDIA official website](https://developer.nvidia.com/cuda-downloads).
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
See [Change Log](changelog.md)
## Contact

If you have any questions or suggestions or want to contribute to the project, open an [issue](https://github.com/wangkingh/FastXC/issues) or submit a pull request.

For more direct inquiries, you can reach the author at:  
**Email:** [wkh16@mail.ustc.edu.cn](mailto:wkh16@mail.ustc.edu.cn)

It will be my great pleasure if my code can provide any help for your research!
