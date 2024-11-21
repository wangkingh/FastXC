# FastXC
High Performance Noise Cross-Corelation Computing Code for 9-component Recordings

This program is designed to efficiently compute single/nine-component noise cross-correlation functions (NCFs) from ambient noise data using a high-performance CPU-GPU heterogeneous computing framework. It integrates data preprocessing, accelerated cross-correlation computation, and various stacking techniques (Linear, PWS, tf-PWS), with a particular emphasis on optimizing the computing process using CUDA technology. This significantly enhances processing speed and the signal-to-noise ratio of the data, making it especially suitable for handling large-scale noise datasets.

## Quick Start
cd FastXC


make veryclean


make


python run.py
