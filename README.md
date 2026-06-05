# FastXC

[English](README.en.md) | [文档索引](docs/README.md) | [配置说明](docs/CONFIGURATION.md) | [架构说明](docs/ARCHITECTURE.md) | [输出说明](docs/OUTPUTS.md) | [结果策略](docs/RESULTS.md) | [更新日志](CHANGELOG.md)

本项目的 v2605 公开整理、文档重写和发布打包过程使用了 OpenAI Codex / GPT Pro 辅助。

FastXC 是一套面向 SAC 波形数据的环境噪声互相关计算流程，主要服务
Linux、WSL 和 HPC 场景。Python 层负责配置解析、SAC inventory、路径规划、
阶段调度和索引格式；CUDA/C 后端负责 SAC2SPEC、互相关、PWS 和 TF-PWS 等重
计算阶段。

支持目标是 Linux 或 WSL + NVIDIA CUDA。原生 Windows 构建不支持。

## 获取代码

从 GitHub clone 或解压发布包后，先进入项目根目录：

```bash
git clone https://github.com/wangkingh/FastXC.git FastXC
cd FastXC

# 或者解压发布包
tar -xf FastXC*.tar.gz
cd FastXC*
```

## 环境要求

- Python 3.10 或更新版本。
- Python 依赖：`numpy`、`pandas`、`scipy`、`tqdm`。
- NVIDIA CUDA Toolkit 和可用 GPU。
- GNU Make 以及 C/CUDA 编译工具链。
- Linux 或 WSL。

CUDA 架构默认自动检测。如果自动检测失败，可以手动指定：

```bash
make install ARCH=sm_89
```

## 安装

下面所有命令都假设当前目录是 FastXC 项目根目录，也就是能看到
`README.md`、`Makefile`、`fastxc/`、`native/` 和 `example/` 的目录。

安装时在项目根目录执行：

```bash
# 先进入任意 Python >= 3.10 环境。
make install
fastxc doctor
```

`make install` 会编译所有受支持的原生后端、把可执行文件部署到 Python 包，
并安装可编辑的 `fastxc` 命令。

如果只想编译原生二进制、不安装 Python 包：

```bash
make native-full      # 编译 sac2spec、xc_fast、ncf_pws、ncf_tfpws
make stage-binaries   # 把编译好的二进制复制进 fastxc/bin/<platform>
```

原生二进制会写入 `bin/`，用于 Python 包分发的 staged binaries 会写入
`fastxc/bin/<platform>/`。

## 两种运行方式

FastXC 支持两种等价的命令风格：

```bash
# 源码树风格：适合还没有安装 fastxc 命令时使用。
python -m fastxc.cli doctor config.ini
python -m fastxc.cli prepare config.ini
python -m fastxc.cli run config.ini

# 包化 CLI 风格：执行 make install 或 pip install -e . 后可用。
fastxc doctor config.ini
fastxc prepare config.ini
fastxc run config.ini
```

两种方式调用的是同一个 Python 入口。`fastxc ...` 更短；`python -m
fastxc.cli ...` 更适合直接在源码目录里调试，或者环境还没有完整安装 console
command 的时候使用。

## 工作流

FastXC 围绕可复用 inventory 运行：

```bash
fastxc prepare config.ini
fastxc run config.ini
```

`prepare` 会扫描 SAC 文件、分配 NSL ID、生成 SAC index、筛选允许路径，并写
出 inventory 元数据。`run` 会读取准备好的 inventory，依次执行频谱转换、互
相关、SourcePack 整理、叠加、旋转和可选导出。

## 快速开始

生成配置文件：

```bash
fastxc init -o config.ini
```

修改 `config.ini` 后执行：

```bash
fastxc doctor config.ini
fastxc prepare config.ini
fastxc run config.ini
```

## 内置示例

仓库包含一套小型匿名三分量 SAC 数据：

```text
example/
```

示例目录的详细说明见 [example/README.md](example/README.md)。

安装完成并确认 `fastxc doctor` 正常后，进入示例目录运行示例：

```bash
cd example
fastxc doctor config.ini
fastxc prepare config.ini
fastxc run config.ini
```

示例 workspace 会写入：

```text
workspace
```

可选绘图检查不属于标准计算流程，只用于本地查看结果。继续留在 `example/`
目录中执行。如果当前环境还没有 `matplotlib`，先安装它：

```bash
python -m pip install matplotlib
```

然后可以把旋转后的 RTZ 叠加结果画成按距离偏移的 line plot。默认只叠加
`ZZ`、`ZR`、`RZ` 三个分量，用不同颜色标注，便于查看相位差：

```bash
python plot_rtz_distance_lines.py \
  --workspace workspace \
  --lag-window 20 \
  --output workspace/plots/rtz_linear_zz_zr_rz_distance_lines.png
```

该命令只读取 `stack/rtz_*_sourcepack` 下的 RTZ 结果；如果启用了 PWS 或
TF-PWS，可通过 `--method pws` 或 `--method tfpws` 绘制对应叠加结果。

## 主要配置项

普通用户通常主要修改这些字段：

```ini
[seisarray1]
sac_dir = /path/to/sac/root
pattern = {home}/{station}/{YYYY}/{station}.{YYYY}.{JJJ}.{*}.{component}.{suffix}
sta_list = NONE
component_list = E,N,Z

[time_filter]
time_start = 2017-09-01 00:00:00
time_end = 2017-09-30 01:00:00
time_list = NONE

[geometry]
external_geo_tsv = NONE

[compute]
sac_len = 86400
win_len = 3600
shift_len = AUTO
delta = 0.1
normalize = AUTO
bands = 0.1/2
whiten = AFTER
max_lag = 100
stack_flag = 100
workspace_dir = /path/to/output/workspace

[device]
gpu_list = 0
cpu_workers = 20

[advance.compute]
skip_step = -1
phase_only = False
distance_range = -1/50000
azimuth_range = -1/360
group_pair_mode = all
autocorr_mode = off
async_poll_sec = 5
sourcepack_async_after_xc = True
pre_stack_size = 10
tfpws_band = FULL
tfpws_taper_hz = AUTO

[advance.storage]
unpack_enabled = True
unpack_target = ALL

[debug]
debug = False
```

其中 `pattern` 描述 `sac_dir` 下 SAC 文件的相对路径。它必须从 `{home}` 开
始，并至少包含 `{station}`、`{component}` 或 `{channel}`，以及能解析日期的
字段，例如 `{YYYY}` + `{JJJ}` 或 `{YYYY}` + `{MM}` + `{DD}`。常用通配符：
`{*}` 匹配任意字符串，`{?}` 匹配一个较短的非路径片段；重复出现的同名字段
必须取相同值。更完整的字段和 pattern 规则见
[配置说明](docs/CONFIGURATION.md)。

`sta_list` 是每个 `[seisarrayN]` 或 `[seisarrayN.source]` 自己的台站白名
单；`NONE` 表示这个数据源不按台站筛选。文件格式是一行一个 station code，
空行和 `#` 开头的注释行会被忽略。多个 source 合并到同一组时，每个 source
先按自己的 `sta_list` 过滤，再合并到同一个 group。

常用字段格式：

- `time_start/time_end`：`YYYY-MM-DD HH:MM:SS`。
- `time_list`：一行一个时间，推荐 `YYYY-MM-DD HH:MM:SS`；空行和 `#` 注释
  会被忽略；`NONE` 表示使用 `time_start/time_end`。
- `bands`：空格分隔的 `fmin/fmax` 频带，例如 `0.2/0.4 0.1/0.2`。
- `shift_len`：秒数，或 `AUTO`；`AUTO` 等于 `win_len`。
- `max_lag`：互相关最大延迟，单位秒。
- `stack_flag`：三位 0/1，依次控制 linear/PWS/TF-PWS，例如 `100` 只做
  linear，`111` 三种都做。
- `distance_range`：`min/max`，单位 km；`-1/50000` 基本等于不限制。
- `azimuth_range`：`min/max`，单位度；`-1/360` 基本等于不限制。
- `group_pair_mode`：`intra` 只算同组，`inter` 只算组间，`all` 全部计算。
- `autocorr_mode`：自相关路径开关。`off` 关闭自相关，`include` 在普通台站对
  之外加入自相关，`only` 只保留自相关。

`[geometry].external_geo_tsv` 用于提供外部台站坐标表。`NONE` 表示从 SAC
header 的 `STLA/STLO/STEL` 读取坐标；当 SAC header 缺少坐标、坐标不可靠，
或者同一 station 需要按 `network/location` 区分坐标时，可以把它设为
一个 TSV 文件路径。TSV 至少需要 `station`、`lat`、`lon` 三列，也兼容
`latitude`、`longitude`；可选列包括 `ele`/`elevation`、`network`、
`location`。`group` 是 FastXC 内部的逻辑分组，不参与外部坐标匹配。如果同
时提供 `network` 和 `location`，FastXC 会优先按 `network + station +
location` 匹配；否则按 `station` 匹配。经纬度使用十进制度，西经和南纬写负
数。相对路径按执行 `fastxc` 命令时的当前目录解析；建议使用绝对路径，或在配
置文件所在目录运行。

```text
station	lat	lon	network	location
A7K2	38.499907	-98.603619	KS	00
```

## 输出结构

`fastxc prepare` 会在 `workspace_dir` 下写出：

```text
inventory.meta.json
manifest/sac_index.tsv
path_plan/allowed_paths.tsv
path_plan/nsl_catalog.tsv
```

`fastxc run` 会继续写出：

```text
commands/
filter.txt
stepack/
ncf/
sourcepack/
stack/
stack/rtz_*_sourcepack/
result_ncf/
progress/
log/
```

`commands/` 会保留 Python 实际调用的 native 命令，便于审查和复现。
各阶段产物的用途、是否可清理、以及 SourcePack/stack/rotate 的关系见
[输出说明](docs/OUTPUTS.md)。

## 如何理解中间结果

FastXC 当前主线尽量避免在计算阶段生成海量散 SAC 文件。中间结果通常是：

```text
pack 二进制大文件 + sourcepack_index.tsv 索引
```

SAC2SPEC 先写 `stepack/`，XC 直接读取这些 worker-batch 频谱文件；XC 的互相关
结果写入 `ncf/xcpack/`。随后 SourcePack 阶段把这些记录整理成
`sourcepack/<timestamp>/sourcepack_index.tsv`。这一层 SourcePack 主要是索
引视图，真实互相关 payload 仍在 `ncf/xcpack/*.xcpack`。

叠加和旋转之后的 SourcePack 则是实际物化后的结果。例如
`stack/linearstack_sourcepack/STACK/linearstack.pack` 里存放 linear stack 后
的新 trace，旁边的 `sourcepack_index.tsv` 记录每条 trace 在 pack 中的位置。
PWS 和 TF-PWS 也使用同样的 pack + index 结构，只是由于 GPU worker 并行，可能
会有多个 worker shard pack。

如果需要传统 SAC 文件，用默认开启的 `unpack` 阶段或手动运行 `fastxc unpack`
导出。也就是说，SAC 是输入格式和最终导出格式，而不是 FastXC 后半段的主要工
作格式。

可选工具：

```bash
fastxc sac2dat -I /path/to/sac_dir -O /path/to/dat_dir
fastxc unpack -I /path/to/sourcepack_index.tsv -O /path/to/sac_dir
```

## 文档

更长的项目说明放在 `docs/`：

- [Configuration](docs/CONFIGURATION.md)：INI 字段、路径 pattern 和常见取值。
- [Architecture](docs/ARCHITECTURE.md)：数据流和模块边界。
- [Outputs](docs/OUTPUTS.md)：各阶段输出目录和产物用途。
- [Results](docs/RESULTS.md)：公开仓库中的结果产物和本地输出保留策略。
- [Changelog](CHANGELOG.md)：架构调整、兼容性变化和历史决策。

## 项目目录

```text
fastxc/              Python 包、CLI、配置解析、调度控制
fastxc/inventory/    SAC inventory、源文件匹配、NSL ID、路径规划
fastxc/system/       可执行文件发现、日志配置、模板导出
fastxc/stages/       主流程阶段调度
fastxc/adapters/     原生可执行程序命令适配
fastxc/runtime/      原生子进程执行和进度轮询
fastxc/io/           SAC、SourcePack 等索引/数据格式读写
fastxc/operators/    Python 实现的 SourcePack、stacking、rotation、filter 算子
fastxc/resources/    内置配置模板和静态资源
native/sac2spec/     CUDA SAC2SPEC 后端
native/xc/           CUDA 互相关后端
native/pws/          CUDA PWS 后端
native/tfpws/        CUDA TF-PWS 后端
configs/             公开 smoke 配置
example/             内置示例：配置、匿名数据和绘图脚本
tools/               可选独立工具
docs/                架构和公开项目说明
```

## 排错

检查 Python 配置和原生可执行文件发现情况：

```bash
fastxc doctor config.ini
```

查看原生构建配置：

```bash
make -C native print-config
```

迁移到新机器时，优先检查：

```bash
nvcc --version
nvidia-smi
make -C native print-config
fastxc doctor config.ini
```

## 作者与引用

如果你有问题、建议，或希望参与改进，欢迎在
[GitHub Issues](https://github.com/wangkingh/FastXC/issues) 中讨论。
更深入的问题也可以通过邮件联系作者：
[wkh16@mail.ustc.edu.cn](mailto:wkh16@mail.ustc.edu.cn)。

如果 FastXC 对你的研究有帮助，欢迎引用下列 FastXC 方法论文：

Wang et al. (2025). [High-performance CPU-GPU Heterogeneous Computing Method
for 9-Component Ambient Noise Cross-correlation](https://doi.org/10.1016/j.eqrea.2024.100357).
Earthquake Research Advances.

## 声明与致谢

FastXC 最早由中国地震局地球物理研究所王伟涛老师团队委托中国科学技术大学
计算机学院李会民教授、网络信息中心孙广中老师和吴超老师团队开发；后续优化
与公开整理主要由作者继续完成。

感谢来自中国科学技术大学、中国地震局地球物理研究所、中国地震局预测所、
中国科学院地质与地球物理研究所的同事和朋友在测试、试运行和反馈中提供的
帮助。

本项目 v2605 公开整理、文档重写和发布打包过程使用了 OpenAI Codex / GPT Pro
辅助。早期 README 标题配图由 ChatGPT 生成。

## 参考文献

- Wang et al. (2025). [High-performance CPU-GPU Heterogeneous Computing Method
  for 9-Component Ambient Noise Cross-correlation](https://doi.org/10.1016/j.eqrea.2024.100357).
  Earthquake Research Advances.
- Bensen, G. D., et al. (2007). [Processing seismic ambient noise data to obtain
  reliable broad-band surface wave dispersion measurements](https://doi.org/10.1111/j.1365-246x.2007.03374.x).
  Geophysical Journal International, 169(3), 1239-1260.
- Cupillard, P., et al. (2011). [The one-bit noise correlation: a theory based
  on the concepts of coherent and incoherent noise](https://doi.org/10.1111/j.1365-246X.2010.04923.x).
  Geophysical Journal International, 184(3), 1397-1414.
- Zhang, Y., et al. (2018). [3-D Crustal Shear-Wave Velocity Structure of the
  Taiwan Strait and Fujian, SE China, Revealed by Ambient Noise Tomography](https://doi.org/10.1029/2018JB015938).
  Journal of Geophysical Research: Solid Earth, 123(9), 8016-8031.

## 许可证

FastXC 使用 MIT License。详见 [LICENSE](LICENSE)。
