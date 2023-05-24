*Read this in other languages: [English](README.md), [简体中文](README.zh-CN.md)*

# FastXC Python Library

FastXC is a simple Python library designed to simplify and streamline the processing of seismic cross-correlation tasks. With it, you can handle tasks such as making lists of SAC files, generating spectrum lists, conducting cross-correlation, and performing stacking operations.

FastXC required some of the CUDA-C Pre-processing, Cross-Correlation and stacking programme. You can send me email to get them.

## Installation

You can install FastXC using pip:

```shell
cd dist
pip install FastXC-0.1.0.tar.gz
```
## Quickstart

The following example provides a quick overview of the main functions provided by FastXC:
Note that you should replace the path in this scripts with your own paths.

```python
from FastXC import MakeSacList,
from FastXC import MakeSpectrumList
from FastXC import MakeStackList
from FastXC import MakeStackListList
from FastXC import Sac2SpecCmdGen
from FastXC import SingleListCmdGen
from FastXC import GenStackCmd
import os
import glob

# Create a list of SAC files in a directory
sac_list_job = MakeSacList('/path/to/sac/files')
sac_list_job.pattern = '{kstnm}.{YYYY}.{JJJ}.{HH}{MI}.{kcmpnm}.{suffix}'
sac_list_job.suffix = 'sac'
# Define time period for which files will be selected
sac_list_job.start_time = '2017-08-01 00:00:00'
sac_list_job.end_time = '2017-08-09 00:00:00'
sac_list_job.kcmpnm = 'Z'
# Write matched files to a text file
sac_list_job.write_matched_files('./saclist.txt')

# Generate command for converting SAC files to Spectrum
sac2spec_job = Sac2SpecCmdGen(
    sac_list='./saclist.txt',
    output_directory='/path/to/output/directory',
    window_length=7200)
# Command generation parameters
sac2spec_job.gpu_list = [0, 1, 2]
sac2spec_job.freq_bands = ['0.2/0.5', '0.5/2', '2/4']
sac2spec_job.whiten_type = 'After'
sac2spec_job.normalize_type = 'Runabs'
sac2spec_job.skip_check_npts = False
sac2spec_job.save_by_time = True
# Execute the generated command
os.system(sac2spec_job.generate_command())

# Create a list of Spectrum files in a directory
MakeSpectrumList(spectrum_directory='/path/to/spectrum/files',
                 spectrum_list_directory='./spectrum_list_directory',
                 num_threads=3)

# Generate command for cross-correlation
ncfdir = '/path/to/ncf/directory'
for listfile in glob.glob('./spectrum_list_directory/*'):
    cross_corr_job = SingleListCmdGen(virt_src_list=listfile,
                                      half_cc_length=500,
                                      output_directory=ncfdir,
                                      gpu_id=0)
    cross_corr_job.checkhead = False
    cross_corr_job.do_cross_correlation = True
    cross_corr_job.save_by_pair = True
    # Execute the generated command
    os.system(cross_corr_job.generate_command())

# Create a list of stack files
MakeStackList(ncf_directory='/path/to/ncf/files',
              stack_list_directory='./stack_list_directory',
              num_threads=3)

# Generate command for stacking
stack_cmd_job = GenStackCmd(ncf_list_list='./stacklistlist',
                            output_directory='/path/to/output/directory')
stack_cmd_job.normalize_output = True
# Execute the generated command
os.system(stack_cmd_job.generate_command())
```
# Documentation
For more information and detailed explanations of the different classes and functions, please refer to the official FastXC documentation.
(Waiting to complete)
# Note
1.Please note that the paths provided to the functions in the example are placeholders and should be replaced with the actual paths to your directories. Please also ensure that you have.

2.Please note that sac2spec_mg,specxc_mg and ncfstack_mpi should be installed and add to environment path.
