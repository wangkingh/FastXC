# Changelog (2024-12-10)

1. **Added `sac2spec_stable` and `sac2spec_new` Pre-processing Methods**:  
   - **stable**: Utilizes a time-domain Butterworth filter (less efficient).  
   - **new**: Employs a frequency-domain roll-off cosine window filter (similar to a Gaussian window).

2. **Makefile Enhancements**:  
   Introduced the `-arch` option in the top-level Makefile, allowing users to specify GPU architectures based on their device’s compute capability. For example, with an NVIDIA RTX 4090 (compute capability 8.9), use:
   ```bash
   make ... -arch=sm_89
   ```

   You can compile and run `check_gpu.cu` in the `utils` folder to determine your GPU’s compute capability.

3. **Precision Improvement in `sac2spec`**:  
   The frequency response calculation of the second-order Butterworth filter now uses double precision.  
   **Important Update**: This significantly improves numerical stability when the filter bandwidth or sampling frequency is small.
