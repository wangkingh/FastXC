# FastXC 工具命令

本文说明 `fastxc` 主流程之外的辅助工具。它们适合调试、格式转换、手动导出和
已有 workspace 的补救处理；常规计算仍建议使用：

```bash
fastxc prepare config.ini
fastxc run config.ini
```

## 速查

| 命令 | 作用 | 常见输入 | 常见输出 |
| --- | --- | --- | --- |
| `fastxc extract` | 拆分 BigSAC 文件 | 包含 `.bigsac` 的目录 | 普通 `.sac` 文件 |
| `fastxc sac2dat` | SAC 转文本 DAT | 包含 `.sac` 的目录 | `.dat` 文本文件 |
| `fastxc sourcepack` | 从 XC pack 手动生成 SourcePack 索引 | `ncf/` 或 `ncf/xcpack/` | `sourcepack/<timestamp>/sourcepack_index.tsv` |
| `fastxc unpack` | 从 SourcePack 导出普通 SAC | SourcePack 目录或 `sourcepack_index.tsv` | 普通 `.sac` 文件 |

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

## `fastxc extract`

`extract` 用于把 BigSAC 文件拆成多个普通 SAC 文件。它会递归扫描输入目录下的
`.bigsac` 文件，并按每条 SAC record 写出单独文件。

```bash
fastxc extract \
  -I /path/to/bigsac_dir \
  -O /path/to/extracted_sac
```

参数：

- `-I, --input`：包含 `.bigsac` 文件的目录。
- `-O, --output`：拆分后的 SAC 输出目录。

这个工具主要服务旧结果、调试结果或外部 BigSAC 数据。当前 FastXC 主流程后半段
主要使用 pack + index 格式，最终导出通常优先使用 `fastxc unpack`。

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

## 分布式辅助命令

以下命令服务静态 timestamp 切片和本地/远程任务调度，不属于普通单机用户的最小
流程：

```bash
fastxc plan config.ini -N 8 -O workspace/distributed
fastxc run-plan workspace/distributed/run_plan.tsv -j 2
fastxc collect-plan workspace/distributed/run_plan.tsv
```

简要含义：

- `plan`：按 timestamp 生成多个 task config 和 `run_plan.tsv`。
- `run-plan`：在本机并发运行计划中的 task。
- `collect-plan`：收集各 task 的 SourcePack 索引，生成主 workspace 可用的
  `sourcepack_inputs.txt`。

这组命令更适合 HPC/多节点实验；普通流程使用 `prepare` + `run` 即可。

## 计划中的 `extract-stepack`

当前还没有正式的 `fastxc extract-stepack` 命令。计划中的用途是从 SAC2SPEC 输出
的 `workspace/stepack/` 中，按 `timestamp + station/network/location + component`
抽取一个台站的多 step 频谱，并导出为 `.mat` 或 `.npz`，用于检查 SAC2SPEC 到
XC 之间的中间结果。

预期形态可能类似：

```bash
fastxc extract-stepack \
  --workspace workspace \
  --timestamp 2023.001.0000 \
  --station A7K2 \
  --components E,N,Z \
  -O A7K2.2023.001.0000.stepack.mat
```

这个工具会复用 `stepack/*.tsv` 里的索引信息读取二进制 `.stepack`，不需要重新
运行 SAC2SPEC 或 XC。正式实现后，应在本文档中把本节移动到“当前可用工具”。

## 什么时候用哪个工具

- 想重新导出最终 SAC：用 `fastxc unpack`。
- XC 已完成但 SourcePack 缺失或损坏：用 `fastxc sourcepack` 重建索引。
- 手里有 BigSAC 想拆成普通 SAC：用 `fastxc extract`。
- 想把少量 SAC 变成文本快速查看：用 `fastxc sac2dat`。
- 想检查 SAC2SPEC 产生的单台站频谱：等待或实现 `extract-stepack`。
