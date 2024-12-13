<p align="center">
  <img src="https://private-user-images.githubusercontent.com/38589094/394993541-1550eb2f-23be-4795-a48a-7f1a12144604.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzM5NzM1MTksIm5iZiI6MTczMzk3MzIxOSwicGF0aCI6Ii8zODU4OTA5NC8zOTQ5OTM1NDEtMTU1MGViMmYtMjNiZS00Nzk1LWE0OGEtN2YxYTEyMTQ0NjA0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMTIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjEyVDAzMTMzOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZiZWU3OTQyMDUxYzVmYzgzMGFmY2QwMTI2MTAxZjJhODkxN2RiYzljNzFlNGNkYzRlNzAxMzY3ZmI4ZTAzMmUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.t8WLt2ABoxLwoFi4lLLLOZ1f-c5JYcklrYRsKX4fVbk" alt="广告图片" width="300">
</p>

* Switch Language 切换语言: [English](README.md)[英文], [简体中文](README.zh-CN.md)*[Simplified Chinese]

## FastXC

* ## Table of Contents
- [Project Introduction](#project-introduction)
- [Installation & Requirements](#installation--requirements)
- [Quickstart](#quickstart)
- [Complete Configuration File Explanation](#complete-configuration-file-explanation)
- [Computational Environment Check](#computational-environment-check)
- [FAQ](#faq)
- [Change Log](#change-log)
- [Author Contact Information](#author-contact-information)
- [Acknowledgements](#acknowledgements)
- [References](#references)

# 💡Project Introduction
High Performance Noise Cross-Corelation Computing Code for 9-component Recordings

Using a high-performance CPU-GPU heterogeneous computing framework, this program is designed to efficiently compute single/nine-component noise cross-correlation functions (NCFs) from ambient noise data. It integrates data preprocessing, accelerated cross-correlation computation, and various stacking techniques (Linear, PWS, tf-PWS), particularly optimizing the computing process using CUDA technology. This significantly enhances processing speed and the signal-to-noise ratio of the data, making it especially suitable for handling large-scale noise datasets.


### Program Features 🎉🎉
1. CUDA-accelerated heterogeneous computing
2. Supports computing both single-component and nine-component cross-correlation functions
3. Employs regex-based file retrieval for SAC files, generally eliminating the need for users to rename files
4. Enables cross-correlation calculation between two seismic arrays
5. Integrates PWS and tf-PWS high-SNR stacking methods (requiring sufficient GPU memory) with CUDA acceleration
6. Separates business logic from low-level computation, allowing users familiar with CUDA and C to customize preprocessing and cross-correlation steps

## 🔧Installation & Requirements
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


## 🚀Quickstart
```python
python run.py
```

## 📝Complete Configuration File Explanation

## 🔍Computational Environment Check

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
## ❓FAQ
## 📒Change Log
See [Change Log](changelog.md)
## 📧Author Contact Information

If you have any questions or suggestions or want to contribute to the project, open an [issue](https://github.com/wangkingh/FastXC/issues) or submit a pull request.

For more direct inquiries, you can reach the author at:  
**Email:** [wkh16@mail.ustc.edu.cn](mailto:wkh16@mail.ustc.edu.cn)

It will be my great pleasure if my code can provide any help for your research!

## 🙏Acknowledgements
We extend our sincere gratitude to our colleagues from the University of Science and Technology of China, the Institute of Geophysics, China Earthquake Administration, the Institute of Earthquake Forecasting, China Earthquake Administration, and the Institute of Geology and Geophysics, Chinese Academy of Sciences, for their __significant contributions__ during this program's testing and trial runs!


ChatGPT generated the title illustration.
## 📚References
Wang et al. (2025). "High-performance CPU-GPU Heterogeneous Computing Method for 9-Component Ambient Noise Cross-correlation." Earthquake Research Advances. Under Review.


Bensen, G. D., et al. (2007). ["Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements."](https://dx.doi.org/10.1111/j.1365-246x.2007.03374.x) Geophysical Journal International 169(3): 1239-1260.


Cupillard, P., et al. (2011). ["The one-bit noise correlation: a theory based on the concepts of coherent and incoherent noise."](https://doi.org/10.1111/j.1365-246X.2010.04923.x) Geophysical Journal International 184(3): 1397-1414.


Zhang, Y., et al. (2018). ["3-D Crustal Shear-Wave Velocity Structure of the Taiwan Strait and Fujian, SE China, Revealed by Ambient Noise Tomography." Journal of Geophysical Research: Solid Earth 123(9): 8016-8031.
	Abstract The Taiwan Strait, along with the southeastern continental margin of the Eurasian plate, Fujian in SE China, is not far from the convergent boundary between the Eurasian plate and the Philippine Sea plate.](https://doi.org/10.1029/2018JB015938)
