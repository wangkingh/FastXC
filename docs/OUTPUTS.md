# FastXC 阶段输出说明

FastXC 的所有运行产物都写在 `[compute].workspace_dir` 下。推荐把每次实验、每
个数据集或每组关键参数放到独立 workspace，避免不同运行相互覆盖。

## 总览

典型目录如下：

```text
workspace_dir/
  config.snapshot.ini
  inventory.meta.json
  filter.txt
  commands/
  log/
  progress/
  manifest/
  path_plan/
  stepack/
  ncf/
  sourcepack/
  stack/
  result_ncf/
```

并非每次运行都会出现所有目录。例如用户可以关闭
`[advance.storage].unpack_enabled` 跳过 SAC 导出。

## `prepare`

`fastxc prepare config.ini` 只做数据发现、筛选和路径规划，不启动重计算。

| 路径 | 说明 |
| --- | --- |
| `config.snapshot.ini` | 本次解析后的配置快照，便于复现。 |
| `inventory.meta.json` | inventory 元数据和配置摘要。 |
| `manifest/sac_index.tsv` | 全局 SAC 清单。每行包含 timestamp、NSL、component、SAC 路径等信息。 |
| `manifest/seisarray*/timestamp_index.tsv` | 单个数据源的时间片索引。 |
| `manifest/seisarray*/by_timestamp/*.tsv` | 按时间片拆分的 SAC manifest，供 SAC2SPEC 阶段使用。 |
| `path_plan/nsl_catalog.tsv` | GNSL 节点表，包含 group/network/station/location、坐标和可用分量。 |
| `path_plan/allowed_paths.tsv` | 允许参与互相关的台站路径对。 |
| `path_plan/allowed_path_ids.txt` | 原生 XC 使用的 path id 白名单。 |

如果 `allowed_paths.tsv` 的行数不符合预期，优先检查 `sta_list`、
`external_geo_tsv`、`distance_range`、`azimuth_range`、`group_pair_mode` 和
`autocorr_mode`。

## `run`: SAC2SPEC

SAC2SPEC 会把 SAC 时域数据切窗、预处理并转换为频域 SEGSPEC。

| 路径 | 说明 |
| --- | --- |
| `filter.txt` | 本次频带和预处理参数对应的 filter 描述。 |
| `stepack/w<worker>.b<batch>.stepack` | 按 worker batch 写出的 step-major 频谱二进制。 |
| `stepack/w<worker>.b<batch>.tsv` | 同一 batch 内 timestamp 虚拟切片索引。 |
| `stepack/_SUCCESS` | SAC2SPEC 阶段完成标记。 |
| `progress/sac2spec_progress.tsv` | SAC2SPEC 进度侧写文件。 |

`stepack` 是 XC 的直接输入。二进制 payload 按 `[step][batch_nslc][freq]`
连续存储；TSV sidecar 记录 timestamp、pack path、NSLC 范围、字节范围和
pitch 元数据，供 native XC 拼出虚拟 timestamp 视图。

## `run`: XC

XC 阶段执行互相关，正式流程写入 packed NCF。

| 路径 | 说明 |
| --- | --- |
| `ncf/xcpack/*.xcpack` | 互相关结果二进制 pack。 |
| `ncf/xcpack/*.tsv` | 每个 pack 的记录索引。 |
| `ncf/xcpack/<timestamp>.done` | 单个时间片完成标记。 |
| `ncf/xcpack/_SUCCESS` | XC 阶段完成标记。 |
| `progress/xc_progress.tsv` | native XC 与 side-task 进度。 |

`xcpack` 不是最终人工浏览格式，后续 SourcePack、stack 和 rotate 会继续引用
这些 pack 中的记录。

## `run`: SourcePack

SourcePack 不复制互相关数据本体，而是把 XC pack 中的记录按 source/receiver
和 component 重排成可索引结构。

| 路径 | 说明 |
| --- | --- |
| `sourcepack/<timestamp>/sourcepack_index.tsv` | 单个时间片的 SourcePack 索引。 |
| `sourcepack/<timestamp>/_SUCCESS` | 单个时间片 SourcePack 完成标记。 |
| `sourcepack/_SUCCESS` | SourcePack 阶段完成标记。 |
| `progress/sourcepack_progress.tsv` | SourcePack 构建进度。 |

`sourcepack_index.tsv` 中的 `record_path`、`record_offset`、`bytes` 指向真实
pack 记录；这也是分析脚本和后续叠加读取数据的入口。XC 后的 SourcePack 主要
是对 `ncf/xcpack` 的索引视图，不复制互相关 payload。

## `run`: Stack

叠加阶段读取多个时间片的 SourcePack，生成 stack SourcePack。

| 路径 | 说明 |
| --- | --- |
| `stack/linearstack_sourcepack/STACK/linearstack.pack` | linear stack 结果 pack。 |
| `stack/linearstack_sourcepack/STACK/sourcepack_index.tsv` | linear stack 索引。 |
| `stack/pws_sourcepack/STACK/` | PWS stack 输出，仅当 `stack_flag` 第二位为 `1`。 |
| `stack/tfpws_sourcepack/STACK/` | TF-PWS stack 输出，仅当 `stack_flag` 第三位为 `1`。 |
| `stack/*_sourcepack/manifests/*_inputs.txt` | 参与该叠加的 SourcePack 输入清单。 |

`stack_flag` 三位依次控制 linear/PWS/TF-PWS。例如 `100` 只会产生
`linearstack_sourcepack`。

stack 后的 SourcePack 是物化结果：pack 文件中已经是叠加后的新 trace。
linear stack 当前写一个 `linearstack.pack`；PWS 和 TF-PWS 由 GPU worker 并行
写出，可能包含 `pws.w000.pack`、`tfpws.w000.pack` 这类 worker shard pack，
最终仍通过统一的 `sourcepack_index.tsv` 读取。

## `run`: Rotate

旋转阶段把 ENZ stack 转成 RTZ stack，输出仍然是 SourcePack 结构。

| 路径 | 说明 |
| --- | --- |
| `stack/rtz_linearstack_sourcepack/STACK/rtz_linearstack.pack` | linear stack 的 RTZ 结果 pack。 |
| `stack/rtz_linearstack_sourcepack/STACK/sourcepack_index.tsv` | RTZ linear stack 索引。 |
| `stack/rtz_pws_sourcepack/STACK/` | PWS 的 RTZ 结果，仅当 PWS 启用。 |
| `stack/rtz_tfpws_sourcepack/STACK/` | TF-PWS 的 RTZ 结果，仅当 TF-PWS 启用。 |

可选绘图脚本默认读取这里的 RTZ stack：

```bash
python example/plot_rtz_distance_lines.py \
  --workspace workspace_dir \
  --lag-window 20
```

## `run`: Unpack

`[advance.storage].unpack_enabled = True` 是默认设置。它会把 SourcePack 或
stack 结果导出为传统 SAC 文件；如果只想保留紧凑的 SourcePack 结果，可以关
闭该步骤。默认 `unpack_target = ALL`，会同时导出 stack 和旋转后的结果。

| 路径 | 说明 |
| --- | --- |
| `result_ncf/` | 固定导出目录，即 `workspace_dir/result_ncf`。 |
| `ncf_<method>_<component_frame>/.../*.SAC` | legacy 风格 SAC 结果目录，具体取决于 `unpack_target` 和启用的 stack 方法。 |

导出 SAC 会增加文件数量和磁盘占用；这些产物属于本地 workspace 输出，通常
不应提交到仓库。

## 运行审计文件

| 路径 | 说明 |
| --- | --- |
| `commands/sac2spec.sh` | Python 实际提交给 SAC2SPEC native 后端的命令。 |
| `commands/xc.sh` | Python 实际提交给 XC native 后端的命令。 |
| `log/fastxc-*.log` | Python 层和 native 后端的运行日志。 |
| `progress/*.tsv` | 长任务进度侧写。适合排错或远程运行时检查。 |

这些文件通常很小，建议和结果一起保留，便于复现和定位问题。

## 哪些文件适合提交到 Git

通常只提交源码、配置示例和文档。不要提交大型运行产物：

- `config.snapshot.ini`、`inventory.meta.json`、`filter.txt`
- `manifest/`
- `path_plan/`
- `workspace*/`
- `stepack/`
- `ncf/`
- `sourcepack/`
- `stack/`
- `rotate/`
- `result_ncf/`
- `ncf_<method>_<component_frame>/`
- native 编译产物和 `bin/`
- 生成的 `plots/`
- 私有测试配置、私有台站表和本机路径

这些目录已经在项目 `.gitignore` 中按公开仓库用途忽略。
