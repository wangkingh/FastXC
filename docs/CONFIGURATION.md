# FastXC INI 配置说明

FastXC 使用 INI 文件描述数据源、时间范围、预处理、互相关、叠加、设备和
workspace。推荐先用模板生成配置，再按自己的数据路径修改：

```bash
fastxc init -o config.ini
```

普通用户通常只需要修改主 README 中列出的核心字段；本文档补充完整字段含义、
路径匹配规则和常见取值。

## 通用约定

- 路径可以写绝对路径或相对路径。普通数据和输出路径通常按运行命令所在目录
  解析；可执行文件查找会额外考虑源码树、包内 staged binaries 和 INI 所在目
  录。实际运行前建议用 `fastxc doctor config.ini` 检查。
- `NONE` 表示禁用或不提供文件。
- `AUTO` 表示让 FastXC 自动选择默认实现或资源参数。
- 布尔值可使用 `True/False`、`yes/no`、`1/0`、`on/off`。
- 新配置推荐使用 `[seisarrayN]` 和全局 `[time_filter]`。旧式
  `[array1]`、`[array2]` 仍可读取，但不建议新项目继续使用。

## SAC Path Pattern

`[seisarrayN].pattern` 描述 `sac_dir` 下 SAC 文件的相对路径。FastXC 会递归扫
描 `sac_dir`，把每个文件路径转成相对路径，然后用 `pattern` 解析 station、
component、日期等字段。

一个典型例子。若当前目录是内置示例目录 `example/`：

```ini
[seisarray1]
sac_dir = data
pattern = {home}/{station}/{YYYY}/{station}.{YYYY}.{JJJ}.{*}.{component}.{suffix}
component_list = E,N,Z
```

这个 pattern 可以匹配类似：

```text
data/A7K2/2022/A7K2.2022.137.raw.E.sac
```

常用规则：

- `{home}` 必须出现在开头，表示 `sac_dir` 根目录。
- 必须包含 `{station}`。
- 必须包含 `{component}` 或 `{channel}`；如果使用 `{channel}`，内部也会把
  它映射成 component。
- 必须包含能解析日期的字段。推荐 `{YYYY}` + `{JJJ}`，或 `{YYYY}` +
  `{MM}` + `{DD}`；`{HH}`、`{MI}` 可用于小时/分钟级文件。
- `{network}` 和 `{location}` 可以省略；省略时默认写成 `VV` 和 `00`。
- 重复出现的同名字段必须取相同值。例如
  `{station}/{station}.{YYYY}.{JJJ}.{component}.sac` 会要求目录名和文件名
  里的 station 一致。
- `{*}` 匹配任意字符串，可跨路径分隔符；适合吞掉不关心的中间字段。
- `{?}` 匹配一个较短片段，不跨 `/`，也不匹配点号、空格和下划线。
- Windows 风格反斜杠会在内部归一化为 `/`。

支持的内置字段包括：

```text
YYYY YY MM DD JJJ HH MI
network event station location component channel
sampleF quality locid suffix arrayID
label0 label1 label2 label3 label4 label5 label6 label7 label8 label9
```

如果匹配不到预期文件，优先检查三件事：`sac_dir` 是否正确、`pattern` 是否
从 `{home}` 开始、`component_list` 是否和文件名中的 component/channel 一致。

## `[seisarrayN]`

一个逻辑台阵或测线组。`N` 必须是 `1` 到 `9`，会写入 GNSL 的 group 字段。
同一个 `N` 可以有多个物理数据源，例如 `[seisarray1]` 和
`[seisarray1.network_b]`，它们会合并进同一逻辑 group。

| 字段 | 说明 |
| --- | --- |
| `sac_dir` | SAC 根目录。设置为 `NONE` 时该源不参与扫描。 |
| `pattern` | SAC 路径匹配规则，见上一节。 |
| `sta_list` | 台站白名单文本文件；`NONE` 表示不按台站文件筛选。 |
| `component_list` | 参与计算的分量列表，如 `E,N,Z`。 |

## `[time_filter]`

控制参与计算的时间范围。

| 字段 | 说明 |
| --- | --- |
| `time_start` | 起始时间，格式 `YYYY-MM-DD HH:MM:SS`。 |
| `time_end` | 结束时间，格式 `YYYY-MM-DD HH:MM:SS`。 |
| `time_list` | 时间白名单文本文件；`NONE` 表示使用 `time_start/time_end` 范围。 |

## `[geometry]`

| 字段 | 说明 |
| --- | --- |
| `external_geo_tsv` | 外部台站坐标表；`NONE` 表示从 SAC header 读取。 |

外部表至少需要 `station`、`lat`、`lon` 字段。若提供 `group/network/location`，
FastXC 会优先按完整 GNSL 匹配，否则按 station 匹配。

## `[executables]`

控制原生后端可执行文件发现。普通用户通常保持 `AUTO`。

| 字段 | 说明 |
| --- | --- |
| `executable_root` | 可执行文件根目录；`AUTO` 会依次查找源码树 `bin/`、包内 staged binaries 等位置。 |
| `sac2spec` | SAC2SPEC 后端路径或 `AUTO`。 |
| `xc` | 互相关后端路径或 `AUTO`。 |
| `pws` | PWS 后端路径或 `AUTO`。 |
| `tfpws` | TF-PWS 后端路径或 `AUTO`。 |

## `[preprocess]`

| 字段 | 说明 |
| --- | --- |
| `sac_len` | 单个 SAC 文件期望长度，单位秒。 |
| `win_len` | 单个计算窗口长度，单位秒。 |
| `shift_len` | 相邻窗口移动步长，单位秒。 |
| `delta` | 采样间隔，单位秒。 |
| `normalize` | 时间域归一化。`AUTO` 会根据频带数选择 `RUN-ABS` 或 `RUN-ABS-MF`；也可设为 `OFF`、`RUN-ABS`、`RUN-ABS-MF`、`ONE-BIT`。 |
| `bands` | 频带列表，格式如 `0.2/0.4 0.1/0.2`。 |
| `whiten` | 谱白化位置：`OFF`、`BEFORE`、`AFTER`、`BOTH`。 |
| `output_phase_only` | 是否输出 phase-only SEGSPEC，默认关闭。 |

## `[xcorr]`

| 字段 | 说明 |
| --- | --- |
| `max_lag` | 互相关最大延迟，单位秒。输出长度约为 `2 * max_lag / delta + 1`。 |
| `distance_range` | 允许距离范围，格式 `min/max`，单位 km。`-1/50000` 基本等于不限制。 |
| `azimuth_range` | 允许方位角范围，格式 `min/max`，单位度。 |
| `group_pair_mode` | 组间规则：`intra` 仅同组，`inter` 仅不同组，`all` 同组和组间都计算。 |

## `[advance.preprocess]`

| 字段 | 说明 |
| --- | --- |
| `skip_step` | 调试用窗口筛选。`-1` 表示保留全部窗口；也可写逗号分隔窗口编号。 |

## `[advance.xcorr]`

正式流程固定写 pack 格式。

| 字段 | 说明 |
| --- | --- |
| `write_mode` | 必须为 `PACK`。 |
| `write_segment` | 是否写 legacy segment 输出，正式流程通常为 `False`。 |

## `[advance.xcache]`

| 字段 | 说明 |
| --- | --- |
| `windows_per_xcache` | 每个 xcache shard 包含的窗口数；`AUTO` 表示按时间片集中写入。 |
| `async_after_sac2spec` | SAC2SPEC 每完成一个时间片就异步构建 xcache。 |
| `async_poll_sec` | 异步轮询间隔，单位秒。 |
| `cleanup_timestamp_spack` | xcache 完成后删除对应 `spack_by_timestamp/<timestamp>`，节省空间。 |

## `[advance.sourcepack]`

| 字段 | 说明 |
| --- | --- |
| `enabled` | 是否构建 SourcePack 索引。正式流程建议保持 `True`。 |
| `sort_within_source` | 是否按 source/receiver/component 排序。 |
| `async_after_xc` | XC 每完成一个时间片就异步构建 SourcePack。 |
| `async_poll_sec` | 异步轮询间隔，单位秒。 |

## `[stack]`

| 字段 | 说明 |
| --- | --- |
| `stack_flag` | 三位开关，依次表示 linear/PWS/TF-PWS。例如 `100` 只做 linear，`111` 三种都做。 |
| `sub_stack_size` | 子叠加大小，主要影响 PWS/TF-PWS。 |
| `tfpws_band` | TF-PWS 权重频带；`FULL` 表示全频段。 |
| `tfpws_taper_hz` | TF-PWS 频带 taper 宽度；`AUTO` 自动选择。 |

## `[device]`

| 字段 | 说明 |
| --- | --- |
| `gpu_list` | GPU worker 列表，如 `0`、`0,1`。重复值如 `0,0` 表示同一 GPU 上两个 worker。 |
| `gpu_memory_mib` | 每个 worker 的显存上限 MiB；`AUTO` 表示按空闲显存自动分配。 |
| `cpu_workers` | Python 侧扫描、索引和本地算子的线程数。 |

## `[storage]`

| 字段 | 说明 |
| --- | --- |
| `workspace_dir` | 当前 run 的工作目录。`prepare` 和 `run` 的所有中间产物与结果都写在这里。 |

## `[unpack]`

默认开启，用于在主流程末尾把最终 SourcePack/stack 结果导出为传统 SAC 文
件。如果只想保留紧凑的 SourcePack 结果，可设为 `enabled = False`。

| 字段 | 说明 |
| --- | --- |
| `enabled` | 是否在主流程末尾自动导出 SAC。 |
| `target` | 导出目标：`FINAL`、`STACK`、`ROTATE`、`ALL`。 |
| `output_dir` | 导出目录；`AUTO` 写到 `workspace_dir/unpacked`。 |
| `threads` | 导出线程数；`0` 表示使用 `[device].cpu_workers`。 |

## `[advance]`

| 字段 | 说明 |
| --- | --- |
| `debug` | 调试模式。会倾向更易追踪的路径，并让 native 日志更详细。 |
