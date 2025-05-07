# Changelog (2025-05-07)  (fastxc build-system & source clean-up)
--------------------------------------------------

**Top-level `Makefile`**

* Added release/debug **build modes** and `MODE=par|seq` switch  
  – `all` → parallel Release (default) `debug` → parallel Debug  
  – `MODE=seq` forces directory-by-directory serial build.

* Introduced `PARFLAG`, `OSYNC` auto-detect (GNU make ≥ 4) and
  colourful **banner output** to keep logs readable under `-j`.

* Exported `CC NVCC ARCH CFLAGS NVCCFLAGS` and removed duplicated
  recursive rules / `--no-print-directory`.

* Added `help`, `clean`, `veryclean` targets with consistent echo.

**Sub-directory `Makefile`s**

* Switched to **`MAKELEVEL`-based auto Debug**:  
  `MAKELEVEL==0` → `-O0 -g -G`; otherwise `-O3` Release.

* Replaced `?=` with plain `=` (or `override`) where Debug flags
  must override top-level settings.

* Fixed variable typos (`CFLAG` → `CFLAGS`) and duplicate blocks.

* Unified CUDA flags to use `--generate-line-info` in Release.

**Code fixes eliminating all compiler warnings**

* `arguproc.c`  
  – initialised `inputFile/outputFile` to `NULL`;  
  – added option-parsing guard + `fopen` failure check.

* `gen_ncf_path.c` & `gen_ccfpath.c`  
  – initialised `saveptr = NULL`;  
  – changed all `strncpy(...,255)` → `MAXNAME-1` with explicit
    NUL-termination;  
  – replaced `snprintf(...,8192,…)` by `sizeof(buf)` or enlarged
    local buffers to 1024;  
  – for dynamic buffers used `2*MAXLINE` consistently.

* `read_segspec.c` – removed unused vars `strm/hd/size`.

* `cuda.main.cu` – enlarged `outfile/logbuf` and switched to
  `snprintf(buf,sizeof buf,…)` (two occurrences).

* `gen_ccfpath.c` additional fix: second `snprintf` now passes the
  *caller-allocated* length (`2*MAXLINE`) instead of `sizeof(ptr)`.

* All `CreateDir`/path helpers now use `snprintf(sizeof buf,…)`
  and 1024-byte local arrays.

Result
------

`make` (parallel), `make MODE=seq` (serial), `make debug`, and
directory-level `make` all build **warning-free** with GCC ≥ 7 and
NVCC 12 on both Linux and WSL environments.


# Changelog (2025-03-26)

1. **fix**: Corrected the cudaMemset call to remove the redundant sizeof(cuComplex) multiplier. in src/xc_dual/cuda.main.cu

Previously, the call used current_batch * vec_size * sizeof(cuComplex), leading to an incorrect byte size. Now it correctly uses current_batch * vec_size, ensuring the memset operation covers the intended range.

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
