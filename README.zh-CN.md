<p align="center">
  <img src="./utils/GPU_vs_CPU.png" alt="广告图片" width="300">
</p>

## FastXC
**切换语言** **Switch Language**: [English](README.md)[英语], [简体中文](README.zh-CN.md)[Simplified Chinese]

可以点击项目Issue查看一些目前存在的bug，我正在加班加点改进！！！

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


###  程序特色🎉🎉
1. CUDA 加速异构计算
2. 适配计算单分量/九分量互相关函数
3. 使用正则匹配方法检索sac文件，一般情况下不太需要使用者修改自己的文件命名
4. 支持两个台阵之间的互相关函数计算
5. 内嵌了PWS,tf-PWS两种高信噪比叠加方法（需要GPU有较大内存），并使用CUDA计算
6. 分隔业务部分和底层计算部分，熟悉CUDA和C的用户可以在预处理、互相关的环节进行DIY

## 🔧安装与环境要求 
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


编译结束后，你可以在bin下尝试执行这些可执行程序，查看他们的输出，检查是否编译成功。例如
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
### 开始计算示例数据
修改完配置文件之后，进入主目录`FastXC`，在控制台键入下列命令即可运行示例代码
```bash
python run.py
```
### 查看输出目录
输出文件的位置由用户在配置文件中设置。在测试例子中，输出目录为 ~/FastXC/test_output。运行结束后，输出目录下会产生若干文件和文件夹，例如：
```bash
test_output/
├── butterworth_filter.txt   # 记录滤波器相关参数，包括巴特沃斯滤波器的零极点
├── cmd_list                 # 程序内部调用的命令记录列表
├── dat_list.txt             # 数据文件列表信息,冗余
├── sac_spec_list            # SAC数据对应的频谱信息列表（sac2spec阶段调用）
├── segspec                  # sac2spec阶段输出结果
├── ncf                      # 每个时间段计算地互相关结果，calculate_style=DUAL 情形下没有这个文件夹
├── stack                    # 不同时段互相关结果叠加后的目录
└── xc_list                  # 互相关对列表记录

```
在这些文件与文件夹中，您最需要关注以下两个目录（假设计算结束后会出现这两个文件夹）：

- ncf：存放每个时段的互相关结果文件。该目录包含按时间窗口或台阵对区分的互相关数据结果文件，用于查看分段处理的互相关信息。
- stack：存放叠加后的最终互相关结果文件，即将所有时段的互相关结果进行线性叠加或PWS、tf-PWS叠加处理后得到的最终成果。

如果您的输出目录中暂未出现 ncf 文件夹，请检查您的配置文件和计算参数，确保已正确完成互相关计算步骤。有时程序只会在特定处理阶段生成 ncf 文件夹与相关结果。

总之，ncf 代表分段处理结果，stack 代表叠加后的综合结果，这两个目录中的文件对后续数据分析和解释尤为重要。


## 📝完整配置文件解析 

### 台阵信息配置（SeisArrayInfo）
[SeisArrayInfo] 部分用于指定一个或两个台阵数据存放的文件夹、文件路径命名格式以及需要计算的时间范围。这一部分的参数主要包括 **台阵数据所在路径**、**数据命名模式**、**处理时间范围**。通过正确配置此部分，您可以灵活地检索和匹配数据。
- __sac_dir_1__ 与 __sac_dir_2__：指定两个台阵的连续波形数据存放的 __绝对路径__，如果不是计算两个台阵之间的互相关，将`sac_dir_2`设置为`NONE`。
- __pattern_1__ 与 __pattern_2__: 定义访问台阵数据文件的路径模式，该模式中可以使用以下占位符：
    - `{home}`: 表示台阵数据的根目录，将自动替换为 `sac_dir_1` 或 `sac_dir_2` 的路径。
    - `{YYYY}`: 四位的年份（例如：2020）。
    - `{YY}`: 两位的年份（例如：20，代表2020年的后两位）。
    - `{MM}`: 两位的月份（01-12）。
    - `{DD}`: 两位的日期（01-31）。
    - `{JJJ}`: 儒略日（Julian Day）。指一年中的第几天（001-365或366）。 
    - `{HH}`: 两位的小时（00-23）。
    - `{MI}`: 两位的分钟（00-59）。
    - `{component}`: 表示数据分量，例如地震数据常用 `Z`, `N`, `E` 等标识。
    - `{suffix}`: 表示文件后缀，如 `SAC` 或其他文件格式扩展名。
    - `{*}`: 通配符，用于匹配任意长度的任意字符。
    - __注意事项__:
        - 占位信息在文件名或者路径中均可，但是每一个占位信息只能出现一次
        - 时间信息需要至少精确到天, 台站信息、分量名信息也是 __必须__ 的
        - 冗余的信息可以用{*}指代
        - 文件命中的分割符目前支持点号`.`和下划线`_`
- __start__ 与 __end__ : 指定互相关计数据检索的起止时间范围。格式为 `YYYY-MM-DD HH:MM:SS`
- __sta_list_1__ 与 __sta_list_2__: 用于指定台阵对应的台站列表文件路径（可以是相对于`run.py`的相对路径）。文件里的每一行都是台站名，与`pattern_1/2`中的`station`字段相匹配。
- __component_list_1__ 与 __component_list_2__ : 指定台阵数据计算互相关时所使用的分量。例如`Z`, `N`, `E`。__计算九分量互相关时__，输入的分量名要严格按照`E、N、Z`的顺序（分量名可以不是这个，比如`BH1`）。


### 计算参数配置 (Parameters)
`[Parameters]`部分用于控制互相关计算过程中的各类参数和算法选择。主要包括频带、时频域归一化选项、叠加选项等。

#### 基本计算设置
- **output_dir**：  
  指定存放互相关计算结果的目录（__绝对路径__）。

- **win_len**（单位：秒）  
  设置互相关计算的时间窗口长度。例如：`win_len=7200`表示以2小时为一个计算片段。

- **delta**（单位：秒）  
  数据采样间隔，滤波器设计等关键步骤依赖其精确性。请确保 `delta` 值准确无误。

- **max_lag**（单位：秒）  
  最大时延，决定互相关函数的单侧长度。例如：`max_lag=1000`表示输出互相关长度为2000秒（双侧1000秒）。

- **skip_step**  
  控制连续数据处理时的跳步行为。  
  - `-1` 表示不跳过任何数据段。  
  - 可以设置为类如 `3/4/-1` 的形式以跳过特定时间段。

- **distance_threshold**（单位：千米）  
  仅对距离小于该阈值的台站对计算互相关

    
#### 频域与时间域归一化设置
- **whiten**：  
  频谱白化应用时机，可选 `BEFORE`, `AFTER`, `BOTH`, `OFF`。  
  - `BEFORE`：在时间域归一化前进行白化，适合与`RUN-ABS-MF`搭配处理长周期数据, (Zhang et al., 2018)。  
  - `AFTER`：在时间域归一化后进行白化，(Bensen et al., 2007)。  
  - `BOTH`：前后均白化，较激进的选择。  
  - `OFF`：不进行白化（测试用）。

- **normalize**：  
  时间域归一化类型，可选`RUN-ABS`, `ONE-BIT`, `RUN-ABS-MF`, `OFF`。  
  - `RUN-ABS`：滑动绝对值归一化  
  - `ONE-BIT`：符号函数归一化  
  - `RUN-ABS-MF`：分频带滑动绝对值归一化  
  - `OFF`：不归一化（测试用）  
  *注：一般实际应用都会选择某种归一化方法，提高计算结果的信噪比。*

- **norm_special**：  
  可选 `CUDA` 或 `PYV`。无特殊需求建议使用`CUDA`。`PYV`主要用于测试与CPU版本的预处理阶段。

- **bands**：  
  设置谱白化/归一化的频段（单位：Hz）。可指定一个或多个频段，例如 `0.01/0.02 0.02/0.05`。  
  程序将使用所有指定频段中的最小和最大频率作为处理的拐角频率。


#### 并行和日志选项
- **parallel**（`True`/`False`）  
  是否启用CPU并行处理。

- **cpu_count**  
  并行处理时使用的CPU核心数量（逻辑核数）。例如：`cpu_count=100`表示使用100个核心（实际为100个并行线程）。

- **debug**（`True`/`False`）  
  是否进入调试模式，一般设为`False`即可。

- **log_file_path**  
  指定日志文件的存放路径，用于记录运行过程中的详细信息和调试信息。
  
#### 结果输出与保存控制
- **save_flag**  
  使用四位 `0/1` 标记控制输出类型，位对应顺序为：`线性叠加`、`PWS`、`tf-PWS`、`每段互相关结果`。  
  例如：`save_flag=1001`表示保存线性叠加和每段互相关结果。  
  *注*：`save_flag` 仅在 `calculate_style=DUAL` 下生效。

- **rotate_dir**：  
  九分量互相关旋转选项，可选 `LINEAR`, `PWS`, `TFPWS`，决定最终叠加与旋转处理方式。

- **todat_flag**  
  与 `save_flag` 类似的四位开关，决定哪些结果需转为 `dat` 格式输出（`LINEAR`,`PWS`,`TFPWS`,`RTZ`）。  
  用法类似 `save_flag`。  
  *注*：`todat_flag` 同样受 `calculate_style` 影响。

- **calculate_style**  
  可选 `MULTI` 或 `DUAL`：  
  - `MULTI`：优化读写性能，仅支持线性叠加。  
  - `DUAL`：节省存储空间，并支持 PWS、TF-PWS。若需要 PWS 或 tfPWS，请使用 `DUAL` 模式。。


### 可执行文件配置 (Command)

`[Command]` 部分用于指定程序运行过程中所需的可执行文件路径。通过将相应的可执行文件路径配置在此部分中，用户无需手动输入命令，即可由主程序自动调用这些工具完成数据预处理、互相关计算、叠加与旋转等任务。

#### 参数说明

- **sac2spec**：  
  指向`sac2spec`可执行文件，用于对SAC格式数据进行转换与频谱计算等前处理步骤。

- **xc_multi**：  
  指向`xc_multi_channel`可执行文件，用于对多台同时段数据进行互相关计算。该模式下仅支持线性叠加计算（MULTI模式）。

- **xc_dual**：  
  指向`xc_dual_channel`可执行文件，用于双台不同时段数据互相关处理，支持线性叠加以及PWS、tf-PWS叠加（DUAL模式）。当需要高信噪比叠加方式时可使用该工具。

- **stack**：  
  指向`ncfstack`可执行文件，用于对互相关结果进行线性叠加，提升输出数据的信噪比。

- **rotate**：  
  指向`RotateNCF`可执行文件，用于对九分量互相关结果进行旋转处理，以获得特定方向上的互相关特征。在九分量计算中，通过该程序可实现分量旋转并输出期望的叠加结果。

正确配置这些可执行文件路径后，`run.py`脚本或其他主程序模块将自动调用相应程序完成各阶段处理。


### GPU 信息 (gpu_info)

 `[gpu_info]` 部分用于指定 GPU 计算资源的配置，包括可用 GPU 设备编号、
 每张 GPU 上的任务数量以及 GPU 显存信息。在 CUDA 异构计算框架下，合理
 分配 GPU 资源可显著提升计算效率。

 #### 参数说明

 - **gpu_list**：
   指定可用的 GPU 设备编号，多个编号用逗号分隔。
   例如：
   ```ini
   gpu_list = 0
   ```
   表示只使用编号为0的一张 GPU 卡。   若有多张卡可并行使用：
   ```ini
   gpu_list = 0,1
   ```

 - **gpu_task_num**：
   为 `[gpu_list]` 中的每张 GPU 指定任务数量。
   例如：
   ```ini
   gpu_task_num = 1
   ```
   表示在 GPU 0 上分配 1 个任务进程。
   若有两张卡 (0 与 1)，各分配1个任务：
   ```ini
   gpu_task_num = 1,1
   ```

 - **gpu_mem_info**（单位：GB）：
   指定每张 GPU 的可用显存大小信息，用于优化任务分配和内存管理。
   例如：
   ```ini
   gpu_mem_info = 24
   ```
   表示该 GPU 有 24GB 显存。若有两张 GPU，分别为40GB和24GB：
   ```ini
   gpu_mem_info = 40,24
   ```
 #### 小结

 在 `[gpu_info]` 配置中合理设置 GPU 编号、任务数和显存信息，有助于在
 CUDA 异构计算下高效利用计算资源。建议在配置前通过 `nvidia-smi` 命令
 查看 GPU 设备的编号与显存情况，以便进行针对性设置。




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



**Q2:** 除了算力之外，这个程序还受什么计算设备性能局限？


**A2:** 其实很大程度受限于磁盘性能，计算量其实已经被优化到了极限，但如果磁盘或者磁盘阵列很拉胯，程序效率也不高。（当然应该还是比纯CPU架构高的）


**Q3:** 为什么`cal_type=MULTI`不支持tf-PWS和PWS？


**A3:** 之后会上线新版本支持的。前面开发的时候想当然以为MULTI使用tf-PWS会慢，后面会有相应的修改的。

## 📒修改日志
详见 [change_log](changelog.md)


## 📧作者联系方式
如果您有任何的关于本程序的问题，欢迎打开和提交 [issue](https://github.com/wangkingh/FastXC/issues)。

如果有更深入的问题和讨论，欢迎直接通过邮件联系我
**电子邮箱:** [wkh16@mail.ustc.edu.cn](mailto:wkh16@mail.ustc.edu.cn)

如果我的程序能对您的工作有所帮助，将是我巨大的荣幸！


## 🙏致谢
感谢来自中国科学技术大学，中国地震局地球物理研究所，中国地震局预测所，中国科学院地质与地球研究所的小伙伴们在测试程序和试运行的过程所作出的 __重要贡献__！


标题配图由 ChatGPT生成！


## 📚参考文献
Wang et al. (2025). "High-performance CPU-GPU Heterogeneous Computing Method for 9-Component Ambient Noise Cross-correlation." Earthquake Research Advances. Under Review.


Bensen, G. D., et al. (2007). ["Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements."](https://dx.doi.org/10.1111/j.1365-246x.2007.03374.x) Geophysical Journal International 169(3): 1239-1260.


Cupillard, P., et al. (2011). ["The one-bit noise correlation: a theory based on the concepts of coherent and incoherent noise."](https://doi.org/10.1111/j.1365-246X.2010.04923.x) Geophysical Journal International 184(3): 1397-1414.


Zhang, Y., et al. (2018). ["3-D Crustal Shear-Wave Velocity Structure of the Taiwan Strait and Fujian, SE China, Revealed by Ambient Noise Tomography." Journal of Geophysical Research: Solid Earth 123(9): 8016-8031.
	Abstract The Taiwan Strait, along with the southeastern continental margin of the Eurasian plate, Fujian in SE China, is not far from the convergent boundary between the Eurasian plate and the Philippine Sea plate.](https://doi.org/10.1029/2018JB015938)




