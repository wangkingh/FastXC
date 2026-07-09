# FastXC 工具命令

本文说明 `fastxc` 主流程之外的辅助工具。它们适合调试、格式转换、手动导出和
已有 workspace 的补救处理。常规计算和阶段补跑仍建议使用 README 中的主流程说明：

```bash
fastxc prepare config.ini
fastxc run config.ini
```

## 速查

| 命令 | 作用 | 常见输入 | 常见输出 |
| --- | --- | --- | --- |
| `fastxc sac2dat` | SAC 转文本 DAT | 包含 `.sac` 的目录 | `.dat` 文本文件 |
| `fastxc sourcepack` | 从 XC pack 手动生成 SourcePack 索引 | `ncf/` 或 `ncf/xcpack/` | `sourcepack/<timestamp>/sourcepack_index.tsv` |
| `fastxc unpack` | 从 SourcePack 导出普通 SAC | SourcePack 目录或 `sourcepack_index.tsv` | 普通 `.sac` 文件 |
| `fastxc extract-ncf` | 抽取单条已计算的 NCF SAC 记录 | SourcePack 索引、`ncf/xcpack` 或 workspace | `.sac` 文件 |
| `fastxc plot-rtz-grid` | 绘制 unpack 后的单分量或 3x3 虚拟炮集 | `result_ncf/ncf_*_BHZ` 或 `result_ncf/ncf_*_RTZ` | PNG |
| `fastxc extract-stepack` | 从 StepPack 抽取单台站频谱，可同时绘图 | `workspace/stepack` | `.mat`，可选 PNG |
| `fastxc plot-stepack-mat` | 绘制 StepPack `.mat` 频谱图 | `extract-stepack` 导出的 `.mat` | PNG |

## `fastxc unpack`

`unpack` 把 SourcePack 中的记录导出为传统 SAC 文件。正常 `fastxc run` 的最后
阶段会按 `[advance.storage]` 自动执行 unpack；这个命令主要用于手动导出某一层
SourcePack，或者重新导出已有结果。

```bash
fastxc unpack \
  -I workspace/stack/linearstack_sourcepack/STACK \
  -O workspace/manual_export/ncf_linear \
  -T 4
```

也可以直接指定单个索引文件：

```bash
fastxc unpack \
  -I workspace/stack/rtz_linearstack_sourcepack/STACK/sourcepack_index.tsv \
  -O workspace/manual_export/ncf_linear_RTZ
```

参数：

- `-I, --input`：SourcePack 目录或 `sourcepack_index.tsv`。
- `-O, --output`：导出的 SAC 根目录。
- `-T, --threads`：并行写文件线程数，默认 `1`。

输出文件会按虚拟源和接收台站组织，例如：

```text
manual_export/ncf_linear/VV.AAA/VV.BBB/VV-VV.AAA-BBB.Z-Z.ncf.SAC
```

## `fastxc sourcepack`

`sourcepack` 从 native XC 的 `xcpack` 输出手动构建 SourcePack 索引。主流程中
这个步骤通常自动完成；手动命令适合以下情况：

- XC 已经完成，但 SourcePack 阶段被中断；
- 想从已有 `ncf/xcpack/` 重新生成索引；
- 调试 SourcePack 排序或索引内容。

```bash
fastxc sourcepack \
  -I workspace/ncf \
  -O workspace/sourcepack
```

也可以直接给 `xcpack` 目录：

```bash
fastxc sourcepack \
  -I workspace/ncf/xcpack \
  -O workspace/sourcepack
```

参数：

- `-I, --input`：XC 输出根目录或 `xcpack` 目录。
- `-O, --output`：SourcePack 输出目录。
- `--keep-order`：保留 XC encounter 顺序，不按 source/receiver/component 排序。
- `--sort`：按 source/receiver/component 排序；这是默认行为。

生成结果形如：

```text
workspace/sourcepack/2023.001.0000/sourcepack_index.tsv
workspace/sourcepack/2023.002.0000/sourcepack_index.tsv
```

这些索引指向 `ncf/xcpack/*.xcpack` 中的真实 payload，不会复制互相关数据。

## `fastxc extract-ncf`

`extract-ncf` 用来从 SourcePack 或 native XC pack 索引中复制一条已经计算好的
NCF 记录，不会重新执行 XC。索引行里保存了 source/receiver/component 元数据，
以及 `record_path` 或 `pack_path`、字节偏移和字节数；被选中的这段 bytes 本身就是
一条完整的 SAC record。

```bash
fastxc extract-ncf \
  --workspace workspace \
  --timestamp 20111222T00_00 \
  --source 45002 \
  --receiver 45009 \
  --component-pair BHE-BHZ \
  -O workspace/plots/45002_45009_BHE_BHZ.SAC
```

也可以直接指定索引文件：

```bash
fastxc extract-ncf \
  -I workspace/sourcepack/20111222T00_00/sourcepack_index.tsv \
  --source 45002 \
  --receiver 45009 \
  --component-pair BHE-BHZ \
  -O one_pair.SAC
```

参数：

- `--workspace`：FastXC workspace；优先读取 `sourcepack/<timestamp>/sourcepack_index.tsv`，找不到时回退到 `ncf/xcpack/*.tsv`。
- `-I, --input`：SourcePack 索引/目录、`xcpack` 目录或 XC 输出根目录。
- `--timestamp`：时间片筛选；同时兼容 `:` 和 `_` 两种时间戳写法。
- `--source`, `--receiver`：虚拟源台站和接收台站。
- `--component-pair`：源端-接收端分量对，例如 `BHE-BHZ` 或 `R-Z`。
- `--src-network`, `--rec-network`, `--src-location`, `--rec-location`：可选的 network/location 消歧过滤。
- `--allow-reverse`：也允许匹配 receiver/source 反向记录。
- `--dry-run`：只打印匹配到的 pack 路径、offset 和 bytes，不写 SAC。

当索引中保存的是容器绝对路径时，工具会回退到本地 workspace 的 `ncf/xcpack`
目录；在 Windows 上也会兼容文件名里 `:` 被替换成 `_` 的情况。
## `fastxc sac2dat`

`sac2dat` 把 SAC 文件转换为 DAT 文本，便于快速查看、绘图或交给只接受文本的
外部脚本。

```bash
fastxc sac2dat \
  -I /path/to/sac_dir \
  -O /path/to/dat_dir
```

参数：

- `-I, --input`：包含 `.sac` 文件的目录。
- `-O, --output`：DAT 输出目录。

这个命令适合小规模检查。大规模结果建议保留 SAC 或 SourcePack，避免文本格式放大
存储体积。

## `fastxc plot-rtz-grid`

`plot-rtz-grid` 从已经 unpack 出来的 `result_ncf` SAC 目录中，选择一个虚拟源
并绘制按震中距排列的虚拟炮集。工具会根据文件名中的
`src_component-rec_component.ncf.SAC` 自动判断布局：

- 单分量结果，例如 `BHZ-BHZ` 或 `Z-Z`：绘制一张子图。
- 三分量结果，例如 `R/T/Z x R/T/Z` 或 `E/N/Z x E/N/Z`：绘制 3x3 九宫格。

这适合快速检查单分量 stack，也适合检查旋转后的 RTZ 九分量互相关结果。

```bash
fastxc plot-rtz-grid \
  -I workspace/result_ncf/ncf_linear_RTZ \
  --source 45002 \
  -O workspace/plots/rtz_grid_45002.png
```

单分量结果也使用同一个命令：

```bash
fastxc plot-rtz-grid \
  -I workspace/result_ncf/ncf_linear_BHZ \
  --source 45002 \
  -O workspace/plots/linear_BHZ_45002.png
```

参数：

- `-I, --input`：unpack 后的 SAC 目录，例如 `ncf_linear_BHZ` 或
  `ncf_linear_RTZ`。
- `--source`：虚拟源台站名，也可以给 `NET.STA`。
- `-O, --output`：PNG 输出路径；不指定时写到输入目录下。
- `--receiver`：只绘制指定接收台站；可以重复指定。
- `--lag-window`：零时延两侧绘图窗口，单位秒。
- `--min-distance, --max-distance`：按距离筛选接收台站，单位 km。
- `--max-receivers, --sample-stride`：限制或抽稀绘图台站数量。
- `--scale`：波形横向缩放；默认自动估计。

这个工具只读取已经导出的 SAC 文件，不直接读取 SourcePack。距离优先使用 SAC 头段
`dist`，缺失时用 `gcarc` 近似换算。

## `fastxc extract-stepack`

`extract-stepack` 从 SAC2SPEC 输出的 `workspace/stepack/` 中，按
`timestamp + station/network/location + component` 抽取一个台站的多 step 频谱，
并导出为 MATLAB 可读的 `.mat` 文件，用于检查 SAC2SPEC 到 XC 之间的中间结果。

```bash
fastxc extract-stepack \
  --workspace workspace \
  --timestamp 2023.001.0000 \
  --station A7K2 \
  --components E,N,Z \
  -O workspace/plots/A7K2.2023.001.0000.stepack.mat
```

参数：

- `--workspace`：FastXC workspace；工具会读取其中的 `stepack/*.tsv`。
- `--stepack`：也可以直接指定 stepack 目录或单个 stepack TSV。
- `--timestamp`：要抽取的时间戳。
- `--station`：台站名。
- `--network, --location`：可选的 network/location 过滤。
- `--components`：逗号分隔的分量列表；默认 `ALL`。
- `--component-match`：分量匹配方式，支持 `exact`、`tail`、`auto`。
- `--no-compress`：关闭 `.mat` 压缩。
- `--plot`：导出 `.mat` 后同时生成快速检查 PNG。
- `--plot-output`：PNG 输出路径；不指定时写到 `.mat` 同目录。

`auto` 分量匹配会先尝试精确匹配，再尝试末尾分量匹配。因此 `--components E,N,Z`
可以匹配 `BHE,BHN,BHZ` 这类三分量命名；如果只想抽取某个原始通道，也可以直接写
`--components BHZ`。

如果希望抽取后立刻生成快速检查图，可以加 `--plot`：

```bash
fastxc extract-stepack \
  --workspace workspace \
  --timestamp 2023.001.0000 \
  --station A7K2 \
  --components E,N,Z \
  -O workspace/plots/A7K2.2023.001.0000.stepack.mat \
  --plot \
  --max-frequency 1.0 \
  --db
```

默认 PNG 会写在 `.mat` 旁边，例如
`A7K2.2023.001.0000.stepack.amplitude.png`。也可以用 `--plot-output`
指定路径。`--quantity`、`--db`、`--min-frequency`、`--max-frequency`、
`--smooth-step`、`--smooth-frequency`、`--no-smooth`、`--plot-title` 和
`--dpi` 与 `plot-stepack-mat` 的绘图含义一致。

导出的 `.mat` 主要包含：

- `spectra`：形状为 `component x step x frequency` 的复数频谱。
- `frequency_hz`：频率轴。
- `step_index`、`timestamp`：step 编号和时间戳。
- `components`、`networks`、`stations`、`locations`：对应的台站通道信息。
- `stla`、`stlo`：台站坐标。

## `fastxc plot-stepack-mat`

`plot-stepack-mat` 绘制 `extract-stepack` 导出的 `.mat` 文件，把每个分量的
step-frequency 频谱画成一张图。默认会做轻微高斯平滑，便于检查整体能量结构。

```bash
fastxc plot-stepack-mat \
  -I workspace/plots/A7K2.2023.001.0000.stepack.mat \
  -O workspace/plots/A7K2.2023.001.0000.stepack.png \
  --max-frequency 1.0 \
  --db
```

参数：

- `-I, --input`：`extract-stepack` 导出的 `.mat` 文件。
- `-O, --output`：PNG 输出路径。
- `--quantity`：绘制 `amplitude`、`power`、`phase`、`real` 或 `imag`。
- `--db`：对 amplitude/power 使用 dB 显示。
- `--min-frequency, --max-frequency`：限制频率范围。
- `--smooth-step, --smooth-frequency`：控制 step 方向和频率方向平滑强度。
- `--no-smooth`：关闭平滑。

## 什么时候用哪个工具

- 想重新导出最终 SAC：用 `fastxc unpack`。
- 想抽取单天、单台站对、单分量对的 NCF：用 `fastxc extract-ncf`。
- XC 已完成但 SourcePack 缺失或损坏：用 `fastxc sourcepack` 重建索引。
- 想把少量 SAC 变成文本快速查看：用 `fastxc sac2dat`。
- 想检查单分量或 RTZ/ENZ 九分量虚拟炮集：用 `fastxc plot-rtz-grid`。
- 想检查 SAC2SPEC 产生的单台站频谱：用 `fastxc extract-stepack --plot`；
  已有 `.mat` 时用 `fastxc plot-stepack-mat` 重新绘图。

旧的 BigSAC/extract 工具链已经废弃。当前结果检查优先使用 SourcePack
`unpack`、`plot-rtz-grid` 和 StepPack inspection 工具。
