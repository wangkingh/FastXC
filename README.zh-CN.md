*切换语言 Switch Language: [English](README.md)[英语], [简体中文](README.zh-CN.md)[Simplified Chinese]

## FastXC
高性能九分量互相关函数计算程序


该程序采用高性能的CPU-GPU异构计算框架，专为高效计算 __地震背景噪声__ 数据中的单/九分量噪声互相关函数（NCF）而设计。
程序集成了数据预处理、加速互相关计算和多种叠加技术（线性、PWS、tf-PWS），并特别通过CUDA技术优化了计算流程。这种优化显著提升了处理速度和数据信噪比，非常适合处理大型噪声数据集。

## 安装
## 系统要求
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
### 编译

## 快速开始（跑通示例数据）

## 互相关计算

## 配置文件修改

## 高级选项

## 计算环境
### 初始检查

## 修改日志
详见 

## 作者联系方式
如果您有任何的关于本程序的问题，欢迎打开 [issue](https://github.com/wangkingh/FastXC/issues)。


如果有更深入的问题和讨论，欢迎直接通过邮件联系我
**电子邮箱:** [wkh16@mail.ustc.edu.cn](mailto:wkh16@mail.ustc.edu.cn)

如果我的程序能对您的工作有所帮助，将是我巨大的荣幸！

