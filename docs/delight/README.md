# Delighter 去光照研究资料

整理日期：2026-11-27。对象是本机 `/Applications/Delighter.app`（Agisoft Texture De-Lighter 2.2.2）。目标是弄清它如何去掉纹理上的阴影和曲面明暗，以及怎样把同一套想法用到任意 JPG，得到接近 Diffuse Map 的固有色。

这些笔记是算法结论，不是可运行实现。Photo2Vector 主线仍是非 AI 矢量化；去光照是独立的前置研究。

## 先读哪一份

| 文件 | 内容 | 状态 |
|---|---|---|
| [02_core_algorithms_math_rea.md](02_core_algorithms_math_rea.md) | 三条算子的公式：HDR 颜色桥、HSV 高光、方向光照、阴影比、闭式 matte、CHOLMOD/CG 分流 | 以这份为准 |
| [01_architecture_and_ui_rea.md](01_architecture_and_ui_rea.md) | 双按钮管线、UI 参数、模块地址、对旧猜测的纠偏 | 架构仍有效；颜色桥以 02 为准 |
| [03_2d_jpg_diffuse_sota_and_adaptation.md](03_2d_jpg_diffuse_sota_and_adaptation.md) | 任意 JPG 的迁移蓝图、2024–2026 模型横评、本机 Intel iMac 算力约束 | 实施建议 |
| [delighting_debug_strings.txt](delighting_debug_strings.txt) | 二进制里的进度串和调试图文件名 | 证据索引 |
| [early/](early/) | 2025-11 只看许可证时写的两篇笔记 | 已被上面三份取代 |

## 已经能当结论用的几点

观测在工作颜色里写成 \(\mathbf{I}=\mathbf{A}\odot\mathbf{S}_{\mathrm{eff}}\)，\(\mathbf{S}_{\mathrm{eff}}=\mathbf{1}+\tilde m(\mathbf{S}-\mathbf{1})\)，反照率是逐通道除法后再夹到 \([0,1]\)。

界面上的两个按钮不是同一条流水线：

- **Remove Shading** 去掉曲面体积光和 AO，依赖网格法线、\(32\times 16\) 经纬分桶和引导滤波。没有 Mesh 的 JPG 搬不走这条。
- **Remove Cast Shadows** 去掉投射本影和半影，依赖亮部/阴影笔触、边界 RGB 光照比、闭式 matte 和弹簧插值。任意 JPG 可以先做这一条。

HDR 进出桥是 \((\mathrm{half}(I/0.36))^{1/2.2}\)，还原是 \(0.36\cdot I^{2.2}\)。\(1/0.36\) 是乘数，不是伽马指数。长边不超过 1024 的修补走 CHOLMOD，更长则缩小后走共轭梯度。`libpoisson6` 是三维点云重建，去光照不用它。

本机没有 CUDA / MPS。扩散类 albedo 模型不作为落地路径。第一版原型应是：两笔涂抹 + Scale-Strip + 闭式 matte + 稀疏求解。

## 没有放进仓库的本地文件

`Desktop/cursor/delight-info/artifacts/` 里还有约 53MB 的 `Delighter.arm64` 薄切片，以及 `disasm/*.s` 反汇编。那是商业程序本体和指令转储，只留在本机，不提交。
