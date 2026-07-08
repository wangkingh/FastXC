# 结果策略

FastXC 生成的结果目录默认不纳入公开仓库。

这些结果目录可能包含本机路径、大型二进制 pack，以及由数据生成的产物。需要公开
验证时，建议只提交清理过的摘要或小型示例，不要提交真实 workspace。

公开验证可以使用内置 smoke 配置：

```bash
fastxc doctor configs/test_suite/public_smoke_1hz_kansas.ini
fastxc prepare configs/test_suite/public_smoke_1hz_kansas.ini
fastxc run configs/test_suite/public_smoke_1hz_kansas.ini
```

该 smoke 配置读取 `example/data` 下的内置示例数据，并把被 `.gitignore` 忽略的
workspace 写到 `example/workspace`。

公开绘图辅助脚本位于：

```text
example/plot_rtz_distance_lines.py
```

打包后的 CLI 也提供 `fastxc plot-rtz-grid`，用于绘制 unpack 后的单分量或 3x3
结果 SAC；`fastxc extract-stepack --plot` 可用于检查 StepPack 频谱。项目本地
检查脚本、生成图片和机器相关结果目录应留在公开仓库之外。
