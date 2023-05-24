*其他语言版本: [English](README.md), [简体中文](README.zh-CN.md)*

# FastXC Python 库

FastXC 是一个简单的 Python 库，设计用于简化和互相关任务的处理。通过它，你可以处理如生成 SAC 文件列表，生成频谱列表，执行交叉相关以及进行叠加操作等任务。
FastXC 执行任务的过程中用到了一些CUDA-C互相关程序，可以联系作者获取。 1531051129@qq.com

# 安装
``` shell
cd dist
pip install FastXC-0.1.0.tar.gz
```
# 快速开始

以下例子提供了 FastXC 提供的主要功能的快速概览：
注意在脚本中你应该用你自己的路径替换掉这里的路径。

```python
from FastXC import MakeSacList
from FastXC import MakeSpectrumList
from FastXC import MakeStackList
from FastXC import MakeStackListList
from FastXC import Sac2SpecCmdGen
from FastXC import SingleListCmdGen
from FastXC import GenStackCmd
import os
import glob

# 在指定目录创建 SAC 文件列表
sac_list_job = MakeSacList('/path/to/sac/files')
sac_list_job.pattern = '{kstnm}.{YYYY}.{JJJ}.{HH}{MI}.{kcmpnm}.{suffix}'
sac_list_job.suffix = 'sac'
# 定义选取文件的时间范围
sac_list_job.start_time = '2017-08-01 00:00:00'
sac_list_job.end_time = '2017-08-09 00:00:00'
sac_list_job.kcmpnm = 'Z'
# 检查输入参数（一定要有这一步）
sac_list_job.check_inputs()
# 筛打印选文件的准则
sac_list_job.print_criteria()
# 执行匹配文件
sac_list_job.match_files()
# 将匹配的文件写入文本文件
sac_list_job.write_matched_files('./saclist.txt')

# 生成将 SAC 文件转换为 Spectrum 的命令
sac2spec_job = Sac2SpecCmdGen(
    sac_list='./saclist.txt',
    output_directory='/path/to/output/directory',
    window_length=7200)
# 命令生成参数
sac2spec_job.gpu_list = [0, 1, 2]
sac2spec_job.freq_bands = ['0.2/0.5', '0.5/2', '2/4']
sac2spec_job.whiten_type = 'After'
sac2spec_job.normalize_type = 'Runabs'
sac2spec_job.skip_check_npts = False
sac2spec_job.save_by_time = True
# 检查和配置输入的参数（一定要有这一步）
sac2spec_job.check_input()
# 打印参数
sac2spec_job.print_selected_parameters()
# 执行生成的命令
os.system(sac2spec_job.generate_command())

# 在指定目录创建 Spectrum 文件列表
MakeSpectrumList(spectrum_directory='/path/to/spectrum/files',
                 spectrum_list_directory='./spectrum_list_directory',
                 num_threads=3)

# 生成交叉相关命令
ncfdir = '/path/to/ncf/directory'
for listfile in glob.glob('./spectrum_list_directory/*'):
    cross_corr_job = SingleListCmdGen(virt_src_list=listfile,
                                      half_cc_length=500,
                                      output_directory=ncfdir,
                                      gpu_id=0)
    cross_corr_job.checkhead = False
    cross_corr_job.do_cross_correlation = True
    cross_corr_job.save_by_pair = True
    # 执行生成的命令
    os.system(cross_corr_job.generate_command())

# 创建叠加文件列表（列表的列表）
MakeStackList(ncf_directory='/path/to/ncf/files',
              stack_list_directory='./stack_list_directory',
              num_threads=3)

# 生成叠加互相关的命令
stack_cmd_job = GenStackCmd(ncf_list_list='./stacklistlist',
                            output_directory='/path/to/output/directory')
stack_cmd_job.normalize_output = True
# 检查和配置输入参数
stack_cmd_job.check_input()
# 系统调用执行该命令
os.system(stack_cmd_job.generate_command())
```

# 文档
如需获取更多信息和对不同类和函数的详细解释，请参阅 FastXC 的官方文档。
(暂时还没有写，可以联系作者询问相关问题)


# 注意事项、
1.请注意，示例中提供给函数的路径是占位符，应用你的实际路径替换。同时，请确保你拥有这些路径。

2.请注意，sac2spec_mg、specxc_mg 和 ncfstack_mpi 应该被安装并添加到环境路径中。


