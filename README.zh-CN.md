<p align="center">
  <img src="https://private-user-images.githubusercontent.com/38589094/394993541-1550eb2f-23be-4795-a48a-7f1a12144604.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzM5NzM1MTksIm5iZiI6MTczMzk3MzIxOSwicGF0aCI6Ii8zODU4OTA5NC8zOTQ5OTM1NDEtMTU1MGViMmYtMjNiZS00Nzk1LWE0OGEtN2YxYTEyMTQ0NjA0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMTIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjEyVDAzMTMzOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZiZWU3OTQyMDUxYzVmYzgzMGFmY2QwMTI2MTAxZjJhODkxN2RiYzljNzFlNGNkYzRlNzAxMzY3ZmI4ZTAzMmUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.t8WLt2ABoxLwoFi4lLLLOZ1f-c5JYcklrYRsKX4fVbk" alt="广告图片" width="300">
</p>

*切换语言 Switch Language: [English](README.md)[英语], [简体中文](README.zh-CN.md)[Simplified Chinese]

## 目录
- [项目介绍](#项目介绍)
- [安装与环境要求](#安装与环境要求)
- [快速开始](#快速开始)
- [完整配置文件解析](#完整配置文件解析)
- [计算环境检查](#计算环境)
- [常见问题 （FAQ)](#常见问题faq)
- [修改日志](#修改日志)
- [作者联系方式](#作者联系方式)
- [致谢](#致谢)
- [参考文献](#参考文献)
## 💡项目介绍
高性能九分量互相关函数计算程序

该程序采用高性能的CPU-GPU异构计算框架，专为高效计算 __地震背景噪声__ 数据中的单/九分量噪声互相关函数（NCF）而设计。
程序集成了数据预处理、加速互相关计算和多种叠加技术（线性、PWS、tf-PWS），并特别通过CUDA技术优化了计算流程。这种优化显著提升了处理速度和数据信噪比，非常适合处理大型噪声数据集。


### 🎉🎉 程序特色
1. CUDA 加速异构计算
2. 适配计算单分量/九分量互相关函数
3. 使用正则匹配方法检索sac文件，一般情况下不太需要使用者修改自己的文件命名
4. 支持两个台阵之间的互相关函数计算
5. 内嵌了PWS,tf-PWS两种高信噪比叠加方法（需要GPU有较大内存），并使用CUDA计算
6. 分隔业务部分和底层计算部分，熟悉CUDA和C的用户可以在预处理、互相关的环节进行DIY

## ⚙️ 安装与环境要求
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


## 🚀快速开始
`FastXC`文件下会有一系列子文件夹和文件，其中最主要的是这5个
```bash
FastXC/src #CUDA,C源代码
FastXC/bin #存放CUDA-C,C写的可执行文件
FastXC/fastxc #调用可执行文件的Python程序
FastXC/config/test.ini #示例计算的配置文件
FastXC/run.py # "主"程序
```
### 修改配置文件
使用`vim`或者其他文件编辑工具修改`FastXC/confist/test.ini`,关于更多配置文件说明详见[完整配置文件解析](#完整配置文件解析)


修改第`5`行：
```ini
sac_dir = /mnt/c/Uers/admin/Desktop/FastXC/test_data
```
为示例数据在您的操作系统中的 __绝对路径__


修改第`27`行：
```ini
output_dir = /mnt/c/Users/admin/Desktop/FastXC/test_output
```
为示例输出在的操作系统中的 __绝对路径__

修改第`84-88`行：
```ini
sac2spec = /mnt/c/Users/admin/Desktop/FastXC/bin/sac2spec
xc_multi = /mnt/c/Users/admin/Desktop/FastXC/bin/xc_multi_channel
xc_dual = /mnt/c/Users/admin/Desktop/FastXC/bin/xc_dual_channel
stack = /mnt/c/Users/admin/Desktop/FastXC/bin/ncfstack
rotate = /mnt/c/Users/admin/Desktop/FastXC/bin/RotateNCF
```
为`FastXC/bin`下这5个程序对应的可执行文件的绝对路径，类似于配置系统环境变量。

修改第`94-96`行，
```ini
gpu_list = 0
gpu_task_num = 1
gpu_mem_info = 24
```
为您打算使用的计算设备的配置信息，您可以使用`nvidia-smi`来查看您的计算设备信息，如果您有两张GPU计算设备，编号分别为`0`,`1`；两张卡的内存（MEMORY）都为40GB；每张卡上部署一个计算任务（ __建议如此__ ），那么你可以改为下面的形式：
```ini
gpu_list = 0,1
gpu_task_num = 1,1
gpu_mem_info = 40,40
```
如果你不熟悉GPU的信息含义，可以查看[计算环境](#计算环境)。
### 开始计算
修改完配置文件之后，进入主目录`FastXC`，在控制台键入下列命令即可运行示例代码
```bash
python run.py
```

## 📝完整配置文件解析

### 台阵信息配置（SeisArrayInfo）
### 计算参数配置 (Parameters)
### 可执行文件配置 (Command)
### GPU 信息 (gpu_info)

## 🔍计算环境
为了运行本CUDA代码，请确保您的工作环境满足以下需求：
1. **英伟达 GPU**:你的计算机（服务器）必须含有支持CUDA（Compute Unified Device Architecture, 统一计算框架111）的GPU设备，通常都是英伟达（Nvidia）公司的GPU。
2. ** CUDA Tookit**: 您需要预先安装好CUDA计算套件，最新的版本可以从英伟达官网上下载 [英伟达官方网站](https://developer.nvidia.com/cuda-downloads)。
3. ** GPU 驱动 （GPU Drivers）**: 请您确保更新英伟达GPU驱动，并且该驱动与CUDA版本相匹配。
### 检查
在开始计算之前，我建议您使用下面这些命令来确认运行环境：、

- 使用`nvidia-smi` 命令检查GPU以及驱动的状态,这个命令会显示GPU状态以及驱动版本:
```bash
nvidia-smi
```

使用下面的命令来查看编译器路径以及确认编译器状态
```bash
which nvcc
nvcc --version
```


## ❓常见问题faq
**Q1:** 该程序是否支持 Windows 环境？ 


**A1:** 原则上该程序主要针对 Linux 环境进行优化，但你尝试使用 WSL 在 Windows 下运行。



**Q2:** 除了算力之外，这个程序还有什么能力局限吗？


**A2:** 其实很大程度受限于磁盘性能，计算量其实已经被优化到了极限，但如果磁盘或者磁盘阵列很拉胯，程序效率也不高。（当然应该还是比纯CPU架构高的）


**Q3:** 为什么`cal_type=MULTI`不支持tf-PWS和PWS？


**A3:** 之后会上线新版本支持的。前面开发的时候想当然以为MULTI使用tf-PWS会慢，后面会有相应的修改的。

## 🗒️修改日志
详见 [change_log](changelog.md)


## 📧作者联系方式
如果您有任何的关于本程序的问题，欢迎打开和提交 [issue](https://github.com/wangkingh/FastXC/issues)。

如果有更深入的问题和讨论，欢迎直接通过邮件联系我
**电子邮箱:** [wkh16@mail.ustc.edu.cn](mailto:wkh16@mail.ustc.edu.cn)

如果我的程序能对您的工作有所帮助，将是我巨大的荣幸！


## 🙏致谢
感谢来自中国科学技术大学，中国地震局地球物理研究所，中国地震局预测所，中国科学院地质与地球研究所的小伙伴们在测试程序和试运行的过程所作出的 __重要贡献__！


## 📚参考文献

