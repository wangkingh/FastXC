# 2026-05-28 SAC2SPEC Spack And Xcache Notes

这篇记录不是传统 release changelog，而是这一轮围绕 `sac2spec -> spack -> xcache -> xc`
链路调整的设计心得。它记录的是为什么要改、改动带来了什么，以及后面继续优化时应该守住哪些边界。

## 背景

最初的链路是：

```text
SAC -> native sac2spec -> 大量单个 .SEGSPEC 小文件 -> Python xcache -> native xc
```

这个结构功能上是清楚的，但在真实数据规模下有一个明显问题：`sac2spec` 阶段会产生大量小文件。
对本地文件系统来说它还能接受，但对网络盘、磁盘阵列、共享文件系统来说，小文件写出会放大 metadata
开销，让 GPU 计算已经结束之后还长时间卡在写出阶段。

这一轮的核心判断是：`sac2spec` 不应该继续把 GPU 产物拆成海量小文件。它更适合输出少量大块顺序文件，
然后把排序、重组、审计这些事情交给 Python 侧的 xcache。

## 当前新链路

现在的推荐链路是：

```text
SAC
  -> native sac2spec
  -> worker-sharded .spack + .tsv sidecar
  -> Python xcache
  -> timestamp-level .xcspec + .json + xcspec_index.tsv
  -> native xc
```

职责边界更清楚：

```text
sac2spec:
  负责 GPU 预处理和频谱生成
  负责把原始 SEGSPEC block 顺序 append 到 spack
  不负责最终 XC 输入布局

xcache:
  读取 spack sidecar
  按 timestamp 聚合
  按 nsl_id/component/network/station/location 排序
  生成 step-major .xcspec

xc:
  只读 xcspec_index.tsv、.xcspec、allowed_paths.tsv
  不再理解零散 SEGSPEC 小文件
```

## 关键设计

### 1. Spack 是 native SAC2SPEC 的高吞吐写出格式

`.spack` 不是新的科学数据格式，而是一个顺序 append 容器。每条记录仍然是：

```text
SEGSPEC header + spectrum payload
```

sidecar TSV 记录：

```text
timestamp
worker_id
pack_path
offset
bytes
original_sac_path
original_spec_path
nsl_id
network
station
location
component
nstep
nspec
dt
df
stla
stlo
```

这样 Python 可以通过 `pack_path + offset + bytes` 直接读取每个 SEGSPEC block，不需要物理解包。

### 2. Xcache 是 native XC 的标准输入层

`.xcspec` 是 timestamp 级别的自描述二进制文件：

```text
Header
SourceEntry table
payload[step][file][freq]
```

payload 使用 step-major 布局：

```text
payload[step][file_index][freq]
```

这和新的 XC 计算模型匹配。XC 每次只需要连续读取一个 step：

```text
step_bytes = file_count * nspec * sizeof(complex64)
```

因此 `nstep` 主要影响循环次数，不会让输入显存占用线性膨胀。

### 3. JSON 只做人类可读和 cache 审计

native XC 不依赖 JSON。JSON 的作用是：

```text
debug
cache reuse 判断
manifest 审计
人类检查
```

这条边界很重要，因为 native 端如果开始解析 JSON，主程序会重新变重。

### 4. Allowed paths 继续外置

`.xcspec` 描述当前 timestamp 里有哪些 source 和频谱数据。
`allowed_paths.tsv` 描述哪些 pair 可以计算，以及路径相关的距离和方位信息。

这两个东西不应该混在一起。pair 策略可能改变，但输入频谱不应该因此重打包。

## 测试观察

Hi-net 测试中，spack 明显减少了 GPU 关键路径里的写出负担。

在 `/mnt/g` 上：

```text
旧 .SEGSPEC 小文件写出: 约 78 s
新 spack 写出:         约 33 到 35 s
```

在 `/mnt/f` 上做完整链路基线：

```text
旧链路:
  sac2spec 小文件模式: 25.76 s
  SEGSPEC -> xcache: 133.16 s
  total: 158.92 s

新链路:
  sac2spec spack 模式: 33.82 s
  SPACK -> xcache: 86.39 s
  total: 120.21 s
```

虽然在 `/mnt/f` 上 spack 写出单独看比小文件模式慢一点，但完整链路仍然快了约 24%。
更重要的是，GPU 占用阶段更早结束，这对共享 GPU 和生产调度更友好。

TL-WF 一天样例：

```text
SAC rows: 60
NSL: 20
allowed paths: 190
timestamp shard: 1
file_count: 60
nstep: 24
nspec: 18523

sac2spec spack 模式: 3.77 s
spack -> xcache: 2.96 s
```

样例 workspace：

```text
/mnt/g/fastxc_tlwf_01line_20170806_1day_xcache_spack
```

## 修改心得

这一轮最重要的收获是：性能优化不一定是继续压 CUDA kernel，而是先把数据通路摆正。

`sac2spec` 原先的问题不在科学逻辑，而在输出形态。海量小文件让 GPU 程序的结尾被文件系统拖住。
这对 HPC 是不理想的，因为 GPU 资源昂贵，应该尽快释放；后续的重排、索引、cache 审计可以交给 CPU/Python。

spack 的意义在于把 native 端从“小文件生产者”改成“大块顺序数据生产者”。这更符合高性能计算常见的 I/O
习惯，也更适合网络存储和并行文件系统。

xcache 的意义在于把 XC 的输入约束固定下来。只要 `.xcspec` ABI 稳定，后面 native XC 可以专注计算：

```text
load one step
H2D
accumulate
next step
write CCF
```

这比让 XC 自己遍历零散 SEGSPEC、理解 timestamp、处理 cache、做输入校验要干净得多。

## 需要继续留意

1. `spack` 的目标大小目前应保持可配置。默认 2 GiB 或 4 GiB 都合理，生产环境要通过真实文件系统测试决定。

2. `xcache` 现在仍然有明显 I/O 成本，但它已经从 GPU 关键路径里移出来了。下一步优化应该优先看：
   - 写 `.xcspec` 的吞吐
   - 是否需要减少 JSON 写出
   - 是否需要 timestamp 级流水线
   - 是否需要边生成 xcache 边删除已消费 spack

3. `.xcspec` ABI 要谨慎变化。Header、SourceEntry、payload 布局一旦被 native XC 依赖，就应该按版本演进。

4. `allowed_paths.tsv` 不要塞进 `.xcspec`。路径策略和频谱输入应该继续分离。

5. Debug 兼容路径可以保留，但生产路径应优先使用 spack 和 xcache。

## 2026-05-30 Static Distributed Planning

在单机异步流水线稳定之后，下一步分布式化不需要第一时间引入 MPI。当前数据形态已经按 timestamp
自然分区：

```text
saclist.tsv
  -> spack_by_timestamp
  -> xcache/<timestamp>.xcspec
  -> sourcepack/<timestamp>/sourcepack_index.tsv
```

因此第一版分布式执行可以是静态切片：

```text
prepare:
  仍然在主控 workspace 生成完整 inventory/path_plan/filter

plan:
  按 timestamp 把 sac_index.tsv 切成多个 task sac_index.tsv
  每个 task 生成一个独立 config.ini
  task config 只覆盖 workspace/executables/device/stack_flag/unpack

run-plan:
  本机模拟时直接 subprocess
  未来远程节点时通过 ssh 执行相同 task config

collect-plan:
  收集所有 task workspace/sourcepack/<timestamp>/sourcepack_index.tsv
  写成全局 sourcepack_inputs.txt
```

收集完成后，主 workspace 可以只运行后半段：

```bash
fastxc run config.ini --only LinearStack,PwsStack,TfPwsStack,Rotate,Unpack
```

这样不会重新触发 `Sac2Spec/XCache/CrossCorrelation/SourcePack`。

这个设计的关键边界是：native 程序不知道自己是否运行在分布式任务里。它只看到一个普通的 task
workspace、一个普通的 sac_index.tsv 和一组普通的 executable paths。

默认无资源配置时，FastXC 隐式注册本机资源：

```text
host = localhost
workspace = <plan_dir>/tasks/task_xxxx/workspace
executables = 当前 config 解析后的绝对路径
gpus = 当前 config 的 gpu_list
```

如果提供资源配置，则只要求目标节点已经有可执行文件、workspace 可写、SAC 路径可读。编译和同步不进入
FastXC 主流程，而由 `Makefile.deploy` 或用户自己的集群环境管理：

```text
Makefile.deploy:
  sync-node
  build-node
  check-node

FastXC:
  plan
  run-plan
  collect-plan
```

这让分布式 v1 只改 Python 控制层，不碰 native ABI 和计算核心。后续如果需要真正多节点协同规约，
再考虑按 `path_id` 分片 stack，或者引入 MPI/任务队列。

## 总结

这轮修改是有意义的。它不是简单地换了一种文件后缀，而是把 FastXC 的第一阶段从文件级流水线推进到块级数据通道。

新的形态更接近 HPC 需要的样子：

```text
少文件
顺序 I/O
明确 ABI
GPU 早释放
Python 做重组
native 做计算
```

这条路线值得继续走下去。

## 2026-05-29 一致性检修

本轮把 native 前两步和 FastXC 主流程重新对齐了一遍：

```text
sac2spec spack 模式
  -> spack_by_timestamp/<timestamp>/*.spack + *.tsv
  -> xcache/*.xcspec + xcspec_index.tsv
  -> native xc --write-mode PACK
  -> ncf/xcpack/*.xcpack + *.tsv
  -> sourcepack/<timestamp>/sourcepack_index.tsv
  -> linearstack_sourcepack
  -> rtz_linearstack_sourcepack
  -> fastxc unpack, if legacy SAC files are needed
```

修正点：

1. 当时 FastXC 的 `sac2spec` 命令生成固定带上 `--spack`，避免默认回到小文件输出。
   2026-05-30 后 native SAC2SPEC 已把 spack 作为唯一输出模式，FastXC 不再传这个过期参数。
2. `xcache` 读取 spack TSV 时兼容当前 native 表头，不再强依赖已经删掉的 `original_spec_path` 字段。
3. 新配置默认且只允许 `write_mode = PACK`，旧写出模式不再进入正式流程。
4. XC 命令生成器固定生成 `--write-mode 3`，避免新链路被悄悄绕开。
5. PACK 模式下，linear stack 和 rotate 都保持 sourcepack 输出；需要传统 SAC 文件时走显式 `fastxc unpack`。

验证结果：

```text
spack -> xcache:
  timestamp: 20170806T00:00
  file_count: 60
  nstep: 24
  nspec: 18523

native xc PACK:
  timestamps: 1
  records: 1710

sourcepack -> linearstack_sourcepack -> rtz_linearstack_sourcepack -> unpack:
  sourcepack indexes: 1
  linear records: 1710
  rotate groups: 190
  unpacked SAC files: 1710
  sample npts: 2001
  sample dt: 0.1
  sample RTZ component: RR
```

结论：当前约定已经能形成闭环。2026-05-30 之后如果继续动 native 输出表头，
FastXC 优先同步 `fastxc/io/spack.py` 或 `fastxc/io/sourcepack.py` 的字段映射。

## 2026-05-29 下一轮 FastXC Stage 重构备忘

在 `sac2spec -> spack -> xcache -> xc -> sourcepack -> stack -> rotate` 这条链路稳定之后，FastXC
自身的结构也需要再整理一轮。现在很多职责已经变成同一条数据流水线里的 stage：

```text
prepare
sac2spec
spack -> xcache
xc
xcpack -> sourcepack
linear stack
pws stack
tfpws stack
rotate
unpack
```

这些步骤应该处在统一层级，而不是一部分在 `controller.py`，一部分在 `adapters/`，一部分在
`operators/` 里隐式调用。

建议下一轮改成：

```text
fastxc/
  controller.py          # 很薄，只负责读取 config、调用 workflow

  workflow/
    modes.py             # SKIP / PREPARE / CMD_ONLY / DEPLOY / ALL
    runner.py            # 顺序调度
    stage.py             # Stage 协议或极轻基类

  stages/
    prepare.py
    sac2spec.py
    xcache.py            # spack -> xcache，后续支持监听 timestamp _SUCCESS
    xc.py
    sourcepack.py        # xcpack -> sourcepack，已支持按 timestamp done 异步整理
    linear_stack.py
    pws_stack.py
    tfpws_stack.py
    rotate.py
    unpack.py

  operators/
    xcache/              # xcache 构建、异步 materializer
    sourcepack/          # sourcepack 构建、unpack
    stacking/            # Python linear stack 实现
    rotation/            # Python rotate 实现

  io/
    spack.py             # spack / SEGSPEC source 表
    xcspec.py            # xcspec binary ABI
    sourcepack.py        # sourcepack / xcpack TSV schema

  adapters/
    native_sac2spec.py   # native 命令生成/运行封装
    native_xc.py
    native_pws.py
    native_tfpws.py
```

职责边界：

```text
stages:
  决定这一步什么时候做、读哪里、写哪里、是否异步、是否清理中间文件

operators:
  操作 FastXC 自有格式，例如 xcache/sourcepack/stack/rotate

adapters:
  封装 native 程序命令行，不理解全局业务流程

controller:
  只组装 workflow，不继续堆具体 stage 逻辑
```

这轮重构不要急着做。更合理的顺序是：

1. 先完成并测试 native SAC2SPEC 的 timestamp-local spack 输出。
2. FastXC 验证 `spack_by_timestamp/<timestamp>/_SUCCESS` 语义。
3. 再把 `xcache` stage 改成监听 timestamp `_SUCCESS`，一边生成 `.xcspec` 一边删除已消费的 timestamp spack。
4. 最后再进行 FastXC stage 层重构，把当前 controller 里的流程逻辑下沉到统一 `stages/`。

这次结构调整的目标不是引入重型框架，而是让主流程变成清楚的 stage 串联：

```python
Workflow([
    PrepareStage(cfg),
    Sac2SpecStage(cfg),
    XCacheStage(cfg),
    XCStage(cfg),
    SourcePackStage(cfg),
    LinearStackStage(cfg),
    PwsStackStage(cfg),
    TfPwsStackStage(cfg),
    RotateStage(cfg),
    UnpackStage(cfg),
]).run(modes)
```

这样后续加 SAC2SPEC 后处理异步、sourcepack 后处理异步、清理策略和测试入口时，不会继续把
`controller.py` 变胖。

## 2026-05-29 Timestamp-Local Spack And Async XCache

native SAC2SPEC 已经把 spack 主路径推进到 timestamp-local：

```text
spack_by_timestamp/<timestamp>/w000.p000000.spack
spack_by_timestamp/<timestamp>/w000.p000000.tsv
spack_by_timestamp/<timestamp>/_SUCCESS
spack_by_timestamp/_SUCCESS
```

FastXC 侧对应新增 `AsyncXCacheMaterializer`：

```text
SAC2SPEC running
  -> timestamp _SUCCESS appears
  -> FastXC reads that timestamp TSV
  -> builds xcache/<timestamp>.xcspec
  -> optional cleanup spack_by_timestamp/<timestamp>
```

这一步的目的不是改变 `.xcspec` ABI，而是把原来的：

```text
sac2spec 全部结束 -> 再整体 spack -> xcache
```

变成：

```text
sac2spec 写完一个 timestamp -> xcache 后台立刻消费
```

理想情况下，`spack -> xcache` 的一部分时间会被 SAC2SPEC 的 GPU/IO 时间覆盖掉，同时
`cleanup_timestamp_spack=True` 时可以避免长期保留一份 spack 和一份 xcache 的双倍空间。

当前保留了同步兜底路径：

```text
[xcache]
async_after_sac2spec = True
cleanup_timestamp_spack = True
```

正式流程默认让清道夫异步工作，避免 spack 和 xcache 长时间双份占用空间。
如果调试 SAC2SPEC 输出，可以显式设置 `cleanup_timestamp_spack=False`。

后续清理逻辑拆成了第三个异步角色：

```text
AsyncXCacheMaterializer:
  只负责 spack_by_timestamp/<timestamp> -> xcache/<timestamp>.xcspec
  成功后写 xcache/_cleanup/spack_ready/<timestamp>.ready

AsyncSpackSweeper:
  只监听 .ready marker
  校验目标目录仍在 spack_by_timestamp 下面且有 _SUCCESS
  删除 spack_by_timestamp/<timestamp>
```

这样生成和删除分开。第一版清道夫只清 SAC2SPEC spack，不碰 xcache、xcpack、sourcepack 或 stack 输出。

## 2026-05-30 FastXC Stage Layer

FastXC controller 已经从“巨型流程文件”拆成薄调度器：

```text
fastxc/controller.py
  只负责读取配置、校验 inventory、解析 step mode、顺序运行 stage

fastxc/stages/
  base.py       mode/state/run helpers
  prepare.py    GenerateFilterStage, PrepareInventoryStage
  *.py          Sac2Spec, XCache, XC, SourcePack, Stack, Rotate, Unpack

fastxc/adapters/
  native command wrapper

fastxc/operators/
  xcache/sourcepack/linear_stack/rotate 等 FastXC 自有数据处理

fastxc/runtime/
  子进程执行、命令审查文件、progress.tsv 轮询
```

新的主流程等价于：

```python
prepare_stages = [
    GenerateFilterStage(),
    PrepareInventoryStage(),
]

compute_stages = [
    Sac2SpecStage(),
    XCacheStage(),
    CrossCorrelationStage(),
    SourcePackStage(),
    LinearStackStage(),
    WeightedStackStage("PwsStack", "pws", 1),
    WeightedStackStage("TfPwsStack", "tfpws", 2),
    RotateStage(),
    UnpackStage(),
]
```

这样之后要继续调整某个阶段，只需要改对应 stage，不继续把 `controller.py`
变胖。异步 xcache、spack sweeper、异步 sourcepack 的生命周期也都留在对应阶段附近，
而不是散在 controller 里面。

同时修正了一个接口残留：当前 native SAC2SPEC 已经把 spack 作为唯一输出模式，
FastXC 不再向 `sac2spec` 命令追加过期的 `--spack` 参数。

## 2026-05-30 Stage Split And Pack-Only Formal Flow

Stage layer 继续拆细，正式流程只保留 PACK/sourcepack 主线：

```text
fastxc/stages/
  sac2spec.py    native SAC2SPEC + async xcache/sweeper lifecycle
  xcache.py      spack_by_timestamp -> xcache
  xc.py          native XC + async sourcepack lifecycle
  sourcepack.py  xcpack index materialization
  stack.py       linear / pws / tfpws sourcepack stack
  rotate.py      sourcepack RTZ rotation
  unpack.py      final SourcePack export
  helpers.py     shared target/method helpers
```

正式 `run` 路径删除了旧的 `APPEND`、`AGGREGATE`、BigSAC list-mode stack
和 BigSAC rotate-list 分支。`write_mode` 现在在配置层收紧为 `PACK`，XC
命令固定生成 `--write-mode 3`。旧 BigSAC 相关逻辑只保留在独立工具/兼容导出层，
不再参与主流程调度。

## 2026-05-30 IO Format Extraction

把纯文件格式层从原来的算子实现里抽出到 `fastxc/io`：

```text
fastxc/io/
  sac_binary.py   SAC record helpers
  spack.py        SAC2SPEC spack/SEGSPEC source discovery
  xcspec.py       .xcspec binary ABI constants and pack helpers
  sourcepack.py   xcpack TSV and sourcepack_index.tsv schema helpers
```

对应地，`operators/xcache` 现在只负责把 spack sources 重排成 step-major
`.xcspec`，`operators/sourcepack` 只负责从 XC 输出构建和导出 SourcePack
索引。也就是：`io` 管格式，`operators` 管算子，`stages` 管流程。

## 2026-05-30 Inventory Flattening

把旧的 `inventory/seis_handler/` 压平成更直接的 prepare 层模块：

```text
fastxc/inventory/
  arrays.py    SeisArray：匹配、筛选、分组的外层对象
  files.py     FileMatcher / FileFilter
  patterns.py  路径模板字段、pattern 编译、timestamp 解析
  planner.py   NSL/GNSL 节点、路径筛选、allowed_paths/nsl_catalog
```

`builder.py` 和 `source_scanner.py` 继续作为 inventory 的组织入口。这样
`inventory` 不再绕一层旧的 `seis_handler` 命名，prepare 阶段的职责可以直接
从文件名看出来。

## 2026-05-30 Operators Rename

把 `fastxc/processing/` 改名为 `fastxc/operators/`。这个名字更贴合当前层级：
它不是单纯的数据格式转换，也不是 native GPU compute，而是 FastXC 自有的
Python 算子实现层，包括 xcache 构建、SourcePack 整理、stacking、rotation
和滤波器生成。

新的边界是：

```text
io          只描述二进制/索引格式怎么读写
operators   执行 FastXC 自有数据产物上的算子
stages      决定算子和 native 程序在完整流程里的运行顺序
adapters    只适配 native 命令行
runtime     只管理子进程、命令审查和进度读取
```

## 2026-05-30 Adapters Rename

把 `fastxc/pipeline/` 改名为 `fastxc/adapters/`。现在完整流程顺序已经由
`stages/` 表达，原来的 `pipeline` 目录只剩 native 可执行程序命令生成与部署：

```text
fastxc/adapters/
  sac2spec.py   生成/运行 native sac2spec 命令
  xc.py         生成/运行 native xc_fast 命令
  stack.py      生成/运行 native pws/tfpws 命令
```

这个命名避免了两层“pipeline”概念混在一起：`stages` 是流程，`adapters`
是 Python 配置到 native 命令行之间的适配层。
