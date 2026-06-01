# FastXC

[English](README.en.md) | [文档](docs/README.md) | [配置说明](docs/CONFIGURATION.md) | [输出说明](docs/OUTPUTS.md) | [更新日志](CHANGELOG.md)

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

[preprocess]
sac_len = 86400
win_len = 3600
shift_len = 3600
delta = 0.1
normalize = AUTO
bands = 0.1/2
whiten = AFTER
output_phase_only = False

[xcorr]
max_lag = 100
distance_range = -1/50000
azimuth_range = -1/360
group_pair_mode = all

[device]
gpu_list = 0
cpu_workers = 20

[storage]
workspace_dir = /path/to/output/workspace

[advance.xcache]
windows_per_xcache = AUTO
async_after_sac2spec = True
cleanup_timestamp_spack = True
```

其中 `pattern` 描述 `sac_dir` 下 SAC 文件的相对路径。它必须从 `{home}` 开
始，并至少包含 `{station}`、`{component}` 或 `{channel}`，以及能解析日期的
字段，例如 `{YYYY}` + `{JJJ}` 或 `{YYYY}` + `{MM}` + `{DD}`。常用通配符：
`{*}` 匹配任意字符串，`{?}` 匹配一个较短的非路径片段；重复出现的同名字段
必须取相同值。更完整的字段和 pattern 规则见
[配置说明](docs/CONFIGURATION.md)。

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
spack_by_timestamp/
xcache/
xcache/xcspec_index.tsv
ncf/
sourcepack/
stack/
stack/rtz_*_sourcepack/
progress/
log/
```

`commands/` 会保留 Python 实际调用的 native 命令，便于审查和复现。
各阶段产物的用途、是否可清理、以及 SourcePack/stack/rotate 的关系见
[输出说明](docs/OUTPUTS.md)。

可选工具：

```bash
fastxc sac2dat -I /path/to/sac_dir -O /path/to/dat_dir
fastxc unpack -I /path/to/sourcepack_index.tsv -O /path/to/sac_dir
fastxc inspect-xcache -I /path/to/xcache
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
fastxc/io/           二进制和索引格式读写
fastxc/operators/    Python 实现的 xcache、stacking、rotation、filter 算子
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

## 许可证

FastXC 使用 MIT License。详见 [LICENSE](LICENSE)。
