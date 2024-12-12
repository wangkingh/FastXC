*切换语言 Switch Language: [English](README.md)[英语], [简体中文](README.zh-CN.md)[Simplified Chinese]

## FastXC
高性能九分量互相关函数计算程序


该程序采用高性能的CPU-GPU异构计算框架，专为高效计算 __地震背景噪声__ 数据中的单/九分量噪声互相关函数（NCF）而设计。
程序集成了数据预处理、加速互相关计算和多种叠加技术（线性、PWS、tf-PWS），并特别通过CUDA技术优化了计算流程。这种优化显著提升了处理速度和数据信噪比，非常适合处理大型噪声数据集。

## 安装
### 系统要求
需要Linux 和 显卡 GPU，建议显卡内存在 8G 及以上


如果您是第一次使用CUDA程序，辛苦您先检查以下系统配置环境 [计算环境](#计算环境)。
### Python 版本要求
-**要求**-: Python 版本3.8或更高版本
### 需要的第三方Python库
- **需要的库**:`obspy`, `pandas`, `scipy`,`matplotlib`, `tqdm`, `numpy`.
- 鉴于我们只使用这些库最基础的一些功能，因此我建议您安装这些库的最新版本即可，如果程序环境中已经有这些库，也不可以不更新。在控制台使用下面的命令安装这些库：
```bash
pip install obspy pandas scipy matplotlib tqdm numpy
```
如果您熟悉anaconda的配置程序环境的方法，那就更好了！
### 编译
整个程序的代码分为两部分。较“高级”的Python部分用于分配计算任务、设计滤波器、生成终端可执行命令的部分，大规模的基本使用`C`和`CUDA-C`完成。对于使用`C`或`CUDA-C`完成的那部分代码，我们需要在运行程序之前将他们编译为可执行文件。根据下面的方法进行编译：
```bash
cd FastXC
make veryclean
make
```
如果不是高级计算卡（比如A100），还要麻烦您修改`FastXC/Makefile`文件的`第三行`：
```makefile
export ARCH=SM_89
```
这里涉及到了计算设备（GPU）的计算能力，您可以自行google设备的计算能力。我也在`FastXC/utils`下准备了一个脚本，编译和运行程序`check_gpu`：
```bash
bash compile.sh
./check_gpu
```
我的台式机使用的的是英伟达RTX4090显卡，运行程序后输出信息如下：
```bash
Device Number: 0
 Device  name: NVIDIA GeForce RTX 4090
 Compute capability:8.9
```
我的计算设备的计算能力是8.9，相应的，编译选项`ARCH=sm_89`

编译结束之后，需要您检查`FastXC/bin`,编译生成的各个可执行文件会储存到这个文件夹里面。至少有`RotateNCF`,`extractSegments`,`ncfstack`,`sac2spec`,`xc_dual_channel`,`xc_multi_channel`。


编译，你能可以在bin下执行这个可执行程序，查看他们的输出，检查是否编译成功。例如
```bash
cd FastXC/bin
./sac2spec
```
## 快速开始（跑通示例数据）

## 互相关计算

## 配置文件修改

## 高级选项
### 调整时/频归一化
### 时频域相位加权叠加（tf-pws）和相位加权叠加pws
## 计算环境
### 初始检查

## 常见问题（FAQ）
## 修改日志
详见 

## 作者联系方式
如果您有任何的关于本程序的问题，欢迎打开 [issue](https://github.com/wangkingh/FastXC/issues)。


如果有更深入的问题和讨论，欢迎直接通过邮件联系我
**电子邮箱:** [wkh16@mail.ustc.edu.cn](mailto:wkh16@mail.ustc.edu.cn)

如果我的程序能对您的工作有所帮助，将是我巨大的荣幸！

