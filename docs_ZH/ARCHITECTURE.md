# FastXC 架构说明

FastXC 当前围绕明确的中间数据格式组织流程，而不是依赖大量人类可读的临时文件。

## 主流程

```text
prepare
  -> manifest/sac_index.tsv
  -> path_plan/nsl_catalog.tsv
  -> path_plan/allowed_paths.tsv

sac2spec
  -> stepack/w<worker>.b<batch>.stepack + .tsv

xc
  -> ncf/xcpack/*.xcpack + *.tsv

sourcepack
  -> sourcepack/<timestamp>/sourcepack_index.tsv

stack
  -> stack/<method>_sourcepack/STACK/sourcepack_index.tsv

rotate
  -> stack/rtz_<method>_sourcepack/STACK/sourcepack_index.tsv

unpack
  -> ncf_<method>_<component_frame>/**/*.SAC
```

## 模块边界

```text
inventory/
  SAC 扫描、files_groups 归一化、NSL ID、台站元数据、path ID 和 allowed path 表

adapters/
  Python 到 native 后端的命令构造

stages/
  单个 pipeline 阶段的高层调度

operators/
  FastXC 自有的数据转换：sourcepack、linear stack、rotate

io/
  二进制和索引格式读写

runtime/
  子进程执行、进度轮询、日志和后台任务摘要
```

## 数据格式角色

- `stepack`：SAC2SPEC 的高吞吐输出，也是 native XC 的输入。每个 worker batch
  保存 header、NSLC 表和按 `payload[step][nslc][freq]` 排列的频谱，旁边的 TSV
  描述虚拟 timestamp 切片。
- `xcpack`：native XC 输出的 pack，按 GPU memory 和任务 shard 写出。
- `sourcepack`：按 source/receiver/component 排序的 pack 记录索引。XC 后的
  SourcePack 主要是指向 `ncf/xcpack` 的索引视图；stack 或 rotate 后则是物化
  产物，自己的 pack 文件中包含新计算出的 trace。

正常流程会把这些二进制/索引格式保留到最后显式 `unpack`。传统 SAC 文件是导出
产物，不是后半段 pipeline 的内部工作格式。

## 已废弃兼容路径

当前主线不再使用 BigSAC 拼接，也不再保留旧 BigSAC extract 工具链。native XC
写出 pack/index 记录，stack 阶段读取 SourcePack 索引，最终给人查看的 SAC 只由
`unpack` 产生。

Planner 输入也统一到当前 `[seisarrayN]` 模型：inventory prepare 阶段传入按
group id 组织的 `files_groups` 映射。旧的 `files_group1/files_group2` 兼容 API
以及 `stas1/times1` 这类配置回填字段已经不属于维护中的流程。
