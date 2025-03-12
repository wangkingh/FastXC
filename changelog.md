# Changlog (2025-03-11)
1. **Replaced `strdup` with custom `my_strdup` to ensure portability in case the compilation environment is not POSIX-compliant.**
  - Switched from strtok_r to the reentrant my_strtok .
  - Refactored xc_dual code to remove unused includes and improve clarity.

# Changlog (2025-02-xx)
1. **Fix some problem in Makefile to ensure using specific "-arch=sm_xx" options**
   
# Changelog (2024-12-17)
1. **Added and renamed four preprocessing (`sac2spec`) methods with different filtering strategies**:
  - `sac2spec`: Frequency-domain bandpass Butterworth filter
  - `sac2spec_butter`: Time-domain bandpass Butterworth filter
  - `sac2spec_cos`: Frequency-domain bandpass Butterworth filter with a raised cosine window
  - `sac2spec_super`: Zero-padded frequency-domain bandpass Butterworth filter

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
