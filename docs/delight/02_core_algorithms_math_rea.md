# Agisoft Texture De-Lighter 2.2.2 三大核心算子的指令级数学复原

分析对象是 arm64 薄切片 `delight-info/artifacts/Delighter.arm64`（55,255,120 字节，`__text` 从 `0x100006544` 起）。函数边界来自 `LC_FUNCTION_STARTS`。常量用 `mov`/`movk` 立即数或只读池里的 IEEE-754 位型解码，不依赖 Universal 二进制上的大范围 `objdump`（那条路径会把地址漂到 `__TEXT` 开头）。

本文只写图形学与图像处理：颜色空间、方向光照、引导滤波、阴影 matte、边界光照比、弹簧插值、UV 接缝表。许可证、加密狗、网络和 crash reporter 不在范围内。下面的公式是从控制流和立即数还原的数学模型，不是商业源码逐行誊写。

置信度分两档：

- **钉死**：本轮在薄切片上读到了对应指令或立即数。
- **推断**：调用关系、调试字符串或前序阅读支持这个形状，但本轮没有把那一段循环逐条读完。

注解缓存：`artifacts/disasm/DPPS_first_vmethod.s`、`artifacts/disasm/rgb_hsv_convert.s`、`artifacts/disasm/kernels_raw.s`。

---

## 0. 总公式

交互式去光照把观测写成一张逐通道乘性光照图，再除掉它。投射阴影这条路径把光照图拆成标量 matte 和三通道比例：

\[
\mathbf{I}(p)=\mathbf{A}(p)\odot\mathbf{S}_{\mathrm{eff}}(p),\qquad
\mathbf{S}_{\mathrm{eff}}(p)=\mathbf{1}+\tilde m(p)\,(\mathbf{S}(p)-\mathbf{1})
\]

\[
\mathbf{A}(p)=\operatorname{clamp}\big(\mathbf{I}(p)\oslash\mathbf{S}_{\mathrm{eff}}(p),\,0,\,1\big)
\]

\(\tilde m\in[0,1]\) 是削减过的阴影不透明度，\(\mathbf{S}\) 是边界上反解出来、再调和插值到全图的三通道光照比。曲面明暗（Remove Shading）走另一条比例：局部方向平均色除以主色，夹到 \([0.05,20]\) 后再做同样的除法。两条路径都在工作颜色上操作；只有数据类型码 5、8、11 会在进出时做下面的 HDR↔感知伽马桥。

---

## 1. 子系统一：曲面明暗、HSV 高光和方向光照

### 1.1 HDR 桥：先乘 \(1/0.36\)，再做 \(1/2.2\) 次方

数据类型码 5、8、11 的进入路径在 `TaskRemoveLightingInteractive::exec` 里构造 `0x4031c71c`，交给 OpenMP worker `0x1009ae2c8`。worker 内部的顺序是钉死的：

1. `0x1009ae4b0`：`fmul`，像素乘以调用方传来的 `0x4031c71c`。该位型是 \(1/0.36=2.77777791\)。
2. 紧接着把乘积打成 float16（指数表在 `0x1028508d8`，缺表项时走 `0x101213130`），再按半精度位型展开回 float32。
3. `0x1009ae54c`：`s1 = 0x3ee8ba2e = 0.45454547 = 1/2.2`，`bl 0x102482da8`。这是幂函数桩，幂在 `s1`，底在 `s0`。

离开路径的对称幂在 `0x1009af12c`：`s1 = 0x400ccccd = 2.2`，同一个桩。`0.36 = 0x3eb851ec` 由调用方在 `0x1009aa278` 一带用 `mov`/`movk` 构造，作为还原时的比例。

\[
I_{\mathrm{work}}=\Big(\operatorname{half}\big(I_{\mathrm{in}}/0.36\big)\Big)^{1/2.2}
\]

\[
I_{\mathrm{out}}=0.36\cdot\big(I_{\mathrm{delit}}\big)^{2.2}
\]

`half(·)` 是上面的 float16 往返，会丢掉尾数。worker 窗口里没有看到对乘积做 `[0,1]` 的 `fmin`/`fmax`；夹紧若存在，在调用方。整型纹理不进入这两个 worker，只做线性归一化后直接参与后面的算子。因此高光 sigmoid、引导滤波的 \(\epsilon\) 和阴影阈值，调的是这条感知域，不是线性光。

`0.36` 和 `2.777…` 都不是 rodata 字面量池里的 `float`，只出现在 `movz`/`movk` 里。幂指数是另外两个立即数，不能把 `1/0.36` 写成伽马指数。

### 1.2 DPPS：HSV 里只改 \(V\)

`rgb2hsv`（`0x1011c70cc`）和 `hsv2rgb`（`0x1011c6ee0`）是标准六扇区变换。\(V=\max(R,G,B)\)，\(S=(\max-\min)/\max\)（\(\max=0\) 时 \(S=0\)），\(H\) 用扇区偏移 \(0,2,4\) 再除以 6，输出落在 \([0,1]^3\)。第三分量在 float 像素上的偏移是 `+8`，也就是 \(V\)。

`DPPSMulti_GetHighlightsCorrection`（`0x100348110`–`0x100348164`）不转颜色，直接读两个像素的 \(V\)：

\[
\Delta V = V[x_2]-V[x_1],\qquad
w=\sigma(8\Delta V-2)=\frac{1}{1+\exp(-(8\Delta V-2))}
\]

指令：`fsub` → `fmov #8` → `fmul` → `fmov #-2` → `fadd` → `fneg` → `bl 0x102482304` → `fadd #1` → `fdiv #1`。结果写成 \((w,w,w)\)。转折点在 \(\Delta V=0.25\)，此处 \(w=1/2\)。

`DPPSSingle_GetHighlightsCorrection`（`0x1003482bc`）先对两个 RGB 调用 `rgb2hsv`，再套同一个 sigmoid。

`DPPSMulti_Delight`（`0x100348164`）把掩码 \(m\) 只混进 \(V\)，\(H,S\) 留在源像素上，三分量都夹到 \([0,1]\)：

\[
V=(1-m)\,V_{\mathrm{src}}+m\,V_{\mathrm{alt}}
\]

`DPPSSingle_Delight`（`0x100348374`）在混完之后调用 `hsv2rgb`。\(m\) 来自一张按像素索引的掩码图，不是上面的 \(w\) 本身；高光修正函数把 \(w\) 交给调用方，再由 Delight 使用。

### 1.3 AO 启发式

亮度权重是 double，不是 Rec.709 的 `0.2126/0.7152/0.0722`（那些位型不在切片里）：

| 权重 | 位型地址 |
|---|---|
| \(0.21\) | `0x1027daa70` |
| \(0.72\) | `0x1027daa60`、`0x1027daa78` |
| \(0.07\) | `0x1027daa68`、`0x1027daa80` |

`AOHeuristicRGB`（`0x100348530`–`0x100348680`）：

\[
Y=0.21R+0.72G+0.07B
\]

\[
w_Y=\sigma\big(10(Y[x_1]-Y[x_2])\big)
\]

实现是乘以 \(-10\)，再走 `exp` 与 \(1/(1+\cdot)\)，没有额外的 `fneg`。色相先转 HSV，圆周距离 \(\delta=\min(|h_1-h_2|,1-|h_1-h_2|)\)，然后

\[
w_H=\sigma(4-100\delta)
\]

（乘以 \(-100\) 再加 4，`fneg` 之后进 `exp`）。返回值三个通道相同：

\[
w=\min(w_Y,w_H)
\]

`AOHeuristicSV`（`0x100348244`，120 字节）工作在已经是 HSV 的输入上，\(S\) 和 \(V\) 各自乘 \(-10\) 后走同一个 logistic，结果放在 `s0/s1/s2`。

### 1.4 方向平均色与 `calculateScaleImpl`

`LLAvg` 查询在 `0x10036faa0`。对象 `+0x58` 的 \(L_{\max}\) 不大于 locality 时，只查 locality 这一层。否则用 AO 字节 \(a_0\)（入口 \(x_3\)）在相邻两层之间线性混合：

\[
a=\operatorname{clamp}\big((a_0\oplus 0\mathrm{xff})/256\cdot\kappa,\,0,\,1\big),\qquad
t=a\cdot L_{\max}
\]

\[
\mathbf{C}=(1-f)\,\mathrm{query}(\lfloor t\rfloor)+f\,\mathrm{query}(\min(\lfloor t\rfloor+1,L_{\max}))
\]

\(1/256=0\mathrm{x}3\mathrm{b}800000\)。某一层返回哨兵 \(-1\) 时改用另一层的有效颜色。\(a_0\) 的低 8 位是 AO 时，\(a_0\oplus 0\mathrm{xff}\) 就是 \(255-\mathrm{AO}\)。

每个体素的方向图是 \(32\times 16\)。`calculateScaleImpl`（`0x100370328`）里，方位角和仰角都先除以池中的 \(2\pi\)（`0x1027fe238`），再 `fcvtzs #5`（乘 32）：

- 两参数 libcall（`atan2` 的调用形态）的结果加 \(0.5\) 后截断，夹到 `0..31`，这是 32 档经度。
- 单参数 libcall（`acos` 的调用形态）的结果直接截断，夹到 `0..15`。因为 \(\arccos(n_z)/(2\pi)\cdot 32=\arccos(n_z)/\pi\cdot 16\)，这 16 档就是纬度。

查询返回的 RGB 是分子。`0x100370440` 从主色指针读出分母，`0x100370450` 起做逐通道 `fdiv`：

\[
\mathbf{s}(p)=\mathbf{C}_{\mathrm{query}}(p)\oslash\mathbf{C}_{\mathrm{main}}
\]

主色无效时的日志字符串是 “Bad main color” / “Can't get LLAvg color”。夹紧边界在 `calculateScale`（`0x100368e04` 起）里写死：`0x1003691b8` 存 `0x3d4ccccd = 0.05`，下一条 `0x1003691c4` 存 `0x41a00000 = 20`，然后以 3 个参数 fork 夹紧核。同一对边界在后面多个数据类型副本里重复出现。

\[
\mathbf{s}_{\mathrm{full}}=\operatorname{clamp}(\mathbf{s},\,0.05,\,20)
\]

球面高斯和极点 \(1/\sin^2\theta\) 展宽位于 `LLAvg` 的 filter 构建（`0x10036ce48` 一带）。前序阅读看到了 `_sinf` 和 `_expf`，本轮没有重读那两个核，展宽的具体 \(\sigma\) 仍是推断。

### 1.5 法线的两轮引导滤波

驱动函数 `0x1003fd8d8`。半径由较长边决定。`0x1003fd994` 比较宽和高，`csel x8, x9, x8, lo` 在 `x8<x9` 时留下 `x9`，否则留下 `x8`，得到 \(M=\max(W,H)\)，再至少取 1：

\[
r_{\mathrm{fine}}=\max\big(1,\operatorname{trunc}(0.001\,M)\big),\qquad
r_{\mathrm{coarse}}=\max\big(1,\operatorname{trunc}(0.01\,M)\big)
\]

`0x3a83126f = 0.001`，`0x3c23d70a = 0.01`。

| 调用 | 地址 | `s0`（\(\epsilon\)） | 半径 |
|---|---|---|---|
| 第一轮 | `0x1003fda00` → `0x10041838c` | `0.001` | \(r_{\mathrm{coarse}}\) |
| 第二轮 | `0x1003fda78` → `0x10041838c` | `0.05`（`0x3d4ccccd`） | \(r_{\mathrm{fine}}\) |

两轮的 `x2` 都是驱动函数的第二个图像参数。第二轮的 `x0` 和 `x1` 指向第一轮的输出缓冲。封装器若把 `x0` 同时当作引导图，第二轮就是自引导；调用点本身没有把 `x2` 换成输出。滤波前，\(\alpha=1\) 的法线按 \(L_2\) 单位化，范数低于 `0x1027fe208` 处的 \(10^{-20}\) 时写成 0。两轮之间的进度回调传入 `50.0`。

引导滤波封装器内部的下采样档 \(s\in\{1,2,4\}\) 和 float16 查表在 `0x10041e250`。本轮没有把那 6 KB 读完，这三档是推断。

---

## 2. 子系统二：投射阴影

编排器是 `0x100373ea4`（约 20 KB）。调试字符串给出的阶段顺序是：修补图集 → 烘焙 matte → 两轮引导滤波 → 烘焙阴影掩码 / 膨胀半影 → 修补半影 → 取比例条带 → 两遍比例修补 → 材质锚点 → 高光抑制 → 未知颜色 → 应用比例。下面只把本轮读到指令的步骤写成公式。

### 2.1 `reduceShadowMatting`

包装函数 `0x1003bcbb4`。`v0` 是高阈值 \(h\)，`v1` 是宽度。\(h=0\) 时直接返回。否则

\[
\ell=\max(h-v_1,\,0)
\]

像素循环在 `0x1003bcde8`：

\[
w=\operatorname{clamp}\Big(\frac{m-\ell}{h-\ell},\,0,\,1\Big),\qquad
m'=m\cdot w
\]

\(h\) 和 \(v_1\) 哪个对应 UI 上的哪一档，要看调用点把参数放进 `v0` 还是 `v1`。本轮钉死的是这个斜坡，不是档位名字。

### 2.2 半影膨胀

`dilatePenumbraFast` 本体在 `0x1003bab5c`，OpenMP 内层到 `0x1003bcbb4`。它和 `umbra` / `penumbra` / `lit` 三张掩码、以及 `atlas_max_filter` 处在同一阶段。内层是图集上的邻域膨胀，具体距离模板（4 邻域、8 邻域或跳洪）本轮没有从 3952 字节的 OpenMP 体里抽完。

编排器 `0x100373ea4`–`0x100378ec4` 里没有整数 `udiv` 常数 1000。半影宽度的像素半径因此不能在本报告里写成已经钉死的 \(w_{\mathrm{pen}}\cdot M/1000\)。UI 默认值仍是千分比 2，那是参数表，不是这条循环里的除法。

### 2.3 边界两侧的 RGB 光照比

随机抽边在 `getImageEdges` 的 worker `0x1003fab00`。令 \(b=\max(1,\max(W,H)\gg 11)\)，梯度 \(e\) 用 xorshift（左移 13、右移 7、左移 17）按下表保留：

| 条件 | 保留概率 |
|---|---|
| \(e>0.2\)（`0x3e4ccccd`） | \(1/(2b^2)\) |
| \(e>0.1\)（`0x3dcccccd`） | \(1/(4b^2)\) |
| \(e>0.05\)（`0x3d4ccccd`） | \(1/(8b^2)\) |
| 其余 | 模数约 \(10b^2\) 的 `fcsel` |

Canny 本身在 `0x1003dec80` 起的约 90 KB 函数里，本轮没有整段还原。

比例反演在 `0x1003f68f4`，常数在 `0x1003f68c0`：`255`、`10^{-6}`（`0x358637bd`）、`0.05`、`20`、`1`。两侧颜色先各自除以一个归一化量再乘 255，记为 \(\mathbf{B}\)（分子）和 \(\mathbf{A}\)（分母）。\(m\) 是边缘处读到的 float：

\[
S_c=\frac{B_c/(A_c+10^{-6})-(1-m)}{m+10^{-6}}
\]

\[
\tilde S_c=\operatorname{clamp}(S_c,\,0.05,\,20)
\]

任一通道 \(\tilde S_c>1\) 时，三个通道都改成寄存器 `s8`（`0x1003f69a4` 起的三次 `fcmp #1`、`orr`、`fcsel`）。`s8` 在这段窗口之前被清零的话，样本就作废；清零本身没有落在这 40 条指令里。

这正是成像方程对 \(S\) 的解。令受光侧预测为 \(\hat{\mathbf{I}}\)，阴影侧观测为 \(\mathbf{I}\)，且 \(\mathbf{I}\approx\hat{\mathbf{I}}\odot\big((1-m)\mathbf{1}+m\mathbf{S}\big)\)，则

\[
S_c=\frac{I_c/\hat I_c-(1-m)}{m}
\]

\(10^{-6}\) 加在乘过 255 之后的分母上，数值上可忽略，结构上就是这个反演。\(\tilde S_c>1\) 表示分子侧比分母侧更亮，和“阴影更暗”不一致，样本被丢掉。同一结构的早一份副本在 `0x1003e20fc`：字节先除以 127 再乘 255（`0x42fe0000` 与 `0x437f0000`），\(10^{-6}\) 只加在分母侧，夹紧同样是 \([0.05,20]\)。127 与后面 Cholesky 写回所用的字节域一致。

### 2.4 未知颜色：先过光照门，再进 CIE Lab

核在 `0x1003da650`。每个通道先形成

\[
u_c=1+m(S_c-1)
\]

并要求 \(0<u_c<1\)。通过之后，把一个字节三元组符号扩展，按 `/127\cdot 255` 放大，再除以 \(u_c\)，夹到 255。

随后是 CIE \(L^*a^*b^*\) 的分段立方根。三个标度是钉死的 double：

| 立即数 | 值 | 角色 |
|---|---|---|
| `0x405d000000000000` | 116 | \(L^*=116f(Y)-16\) |
| `0x407f400000000000` | 500 | \(a^*\) |
| `0x4069000000000000` | 200 | \(b^*\) |

每个通道是 `fcsel`：大于阈值走立方根桩 `0x102482d9c`，否则走线性段。栈上的 \(3\times 3\) 是进 Lab 之前的线性变换，九个系数本轮没有逐个抄出，按位置它是 RGB→XYZ。立方根之后 `bl 0x101aebb28`，把 \((L,a,b)\) 送去和受光样本分布做距离。距离如何压回 \(m\)（草稿里的 \(\exp_2(-(d/\tau)^2)\)）本轮没有打开，不写进公式。

### 2.5 应用比例

未知颜色核里的 \(1+m(S-1)\) 就是 \(\mathbf{S}_{\mathrm{eff}}\)。最终除法 \(\mathbf{A}=\operatorname{clamp}(\mathbf{I}\oslash\mathbf{S}_{\mathrm{eff}},0,1)\) 与子系统一的 `image` 除法是同一形状；编排器后段的具体 `fdiv` 地址沿用前序对 `0x100377c00` 一带的阅读，本轮没有重读那一条存储。

高光抑制（分位点反推 \(\delta m\)）只在调试字符串和函数边界上定位过，反推公式未逐指令核对。

---

## 3. 子系统三：matte、弹簧修补、接缝表

### 3.1 PatchMatch matte

这不是补洞。未知像素在前景样本 \(\mathbf{F}\)（阴影色）和背景样本 \(\mathbf{B}\)（受光色）之间估计不透明度。传播核 `0x10042e39c`，代价从 `0x10042e5e4` 起。

三个通道各自 `fcmp gt`（`0x10042e608`、`0x10042e628`、`0x10042e640`）：候选对里较亮的那一侧若在任一通道上更亮，直接拒绝。在“阴影样本不应比受光样本更亮”的约定下，这就是 \(\mathbf{F}\le\mathbf{B}\) 的逐通道门。

通过后门的 \(\alpha\) 是线段投影，\(\epsilon=10^{-6}=0\mathrm{x}358637\mathrm{bd}\)，下界寄存器是 `movi v8, #0`，上界是 `fmov s9, #1`：

\[
\alpha=\operatorname{clamp}\left(
\frac{(\mathbf{F}-\mathbf{B})\cdot(\mathbf{I}-\mathbf{B})}{\|\mathbf{F}-\mathbf{B}\|^2+10^{-6}},\,0,\,1\right)
\]

预测色 \(\hat{\mathbf{I}}=\alpha\mathbf{F}+(1-\alpha)\mathbf{B}\)。代价在 `0x10042e788` 之后：

\[
\mathcal{E}_c=\|\mathbf{I}-\hat{\mathbf{I}}\|_2
\]

\[
\mathcal{C}=\frac{\|\mathbf{X}-\mathbf{X}_F\|}{\sigma_F}+\frac{\|\mathbf{X}-\mathbf{X}_B\|}{\sigma_B}+\frac{\mathcal{E}_c}{\sigma_c}
\]

每个商用 `fmaxnm` 托在 1 上，\(\sigma\) 本身用 \(10^{-6}\) 托底。\(\mathcal{C}\) 小于当前最优才写下新的样本下标和 \(\alpha\)（`0x10042e7ac`）。

样本槽步长：颜色 `0x18`，状态 `0x30`，位置 `0x0c`。传播开始时若样本向量短于 10 个（`cmp #9`），会补到 10。`initSamples` 使用 float 阈值 `127`（`0x42fe0000`）和 `255`（`0x437f0000`）给 trimap 分类。`randomWalk` 在 `0x10042e8cc`，复用同一个 \(10^{-6}\)。指数减半的搜索半径（\(k=0..7\)）出现在该函数的比较里，本轮没有把随机步的取模逐条写完，步数上限按前序阅读保留为推断。

三分图与笔刷的对应（128 与 255 谁是阴影）由烘焙阶段的映射决定。传播核只消费已经写成 0 / 128 / 255 的 trimap。

### 3.2 弹簧系统：长边不超过 1024 走 CHOLMOD，否则金字塔加 CG

调度器 `0x10043fa0c`。`0x10043fbe8` 取出宽和高。`cmp x8, x9` 之后的 `csel x8, x9, x8, lo`（`0x10043fbf0`）在第一边更小时改留第二边，结果是较长边，再和 `0x400` 比较：

```text
m = max(width, height)
if m > 1024:                         # 0x10043fbf4 b.hi 0x10043fd60
    用 0x10048e468 缩小图像和掩码
    递归调用 0x10043fa0c             # 0x1004400f4
    再放大，作为细层初值            # inpaint_x0_
    bl inpaintCG                     # 0x10044029c → 0x100441d08
else:
    bl inpaintCholesky               # 0x10043fcb4 → 0x10044083c
```

2000×512 的长边是 2000，走缩小再递归。800×900 的长边不超过 1024，走 CHOLMOD。

直接法的求解入口是 `CholmodSolverImpl::solve`（`0x101b29e0c`）。该模板把已知像素以有符号字节读入，除以 `127.0`（`0x405fc00000000000`，不是 124），解完乘回 127，夹到 \([0,127]\) 再写回。其它数据类型有平行的模板副本，127 只属于这一份字节实例。弹簧记录步长 `0x90 = 144`。构建弹簧时传入的 double 权重是 `1.0`。

CG 侧在 `0x100442d24` 构造 `0x358637bd = 10^{-6}`，在 `0x100442eb8` 把 `0x38d1b717 = 10^{-4}` 和 `5000`（`0x1388`）存进相邻栈槽，然后进入迭代。正规方程的形状来自这些立即数加上 “AtA nnz” / “Building symmetric matrices”：

\[
(A^\top A+10^{-6}I)\,\mathbf{x}=A^\top\mathbf{b}
\]

每条弹簧的系数是 \(+1,-1\)（权重点是 1），所以 \(A^\top A\) 是图拉普拉斯，\(\mathbf{b}\) 只在弹簧另一端落在已知像素时非零。这是调和插值，右端不是一张引导梯度场。邻域是 4 连通还是带接缝的 4 连通，由 `TeleportMap` 在建边时插入；`w4 = #4` 的那条比较在弹簧构建开头，本轮没有把去重排序整段读完。排除掩码上的像素不建边，效果是这条边界不提供 Dirichlet 值。

半影修补把 fill 放在半影、exclude 放在图集外和本影上，因此 Dirichlet 只来自受光侧。这是编排器字符串和掩码阶段的结构，内层求解器本身只看见 fill / exclude / known 三类像素。

### 3.3 `makeTeleportMap`

函数 `0x1003feeac`。对象 `+0x18` 的计数小于 1 时（`0x1003feefc`）跳到 `0x1003ff1ac`，输出向量被写成 0，循环不跑。没有接缝列表时传送表是空的。

非空时，每条记录 12 字节：从图表里取出三个 `int32`（偏移 0、4、8）推进 vector。`0x5555…5555` / `0xaaab` 是 12 字节元素的容量除法，不是坐标公式。`0x1003ff9e0` 的 `cmp w8, #0x19` 把 UV 平面上的排斥搜索限制在平方距离 25（5 像素）以内。三维近邻的 \(K\) 和“第一个不属于本岛的候选即对岸”这一句，本轮没有从 KD-Tree 调用里重推，保留为推断。

纯 2D、没有网格时，位置图由 `0x100373330` 填成 \((x,y,0,1)\)，接缝列表为空，于是传送表全空。下游邻域查询退化为普通 4/8 邻域。位置图的填充在前序对 `#6710` 的阅读里，本轮钉死的是“计数小于 1 → 空表”。

---

## 4. 调用关系

```text
TaskRemoveLightingInteractive::exec
  数据类型 5/8/11 → gamma worker 0x1009ae2c8
                    乘 1/0.36，float16 往返，pow(·, 1/2.2)
  Remove Shading
    LLAvg 查询 0x10036faa0
    calculateScaleImpl 0x100370328 → 夹紧 [0.05, 20]
    法线引导滤波 0x1003fd8d8 → 0x10041838c（ε = 0.001，再 ε = 0.05）
  DPPS*
    GetHighlightsCorrection 0x100348110 / 0x1003482bc
    Delight 0x100348164 / 0x100348374
    AOHeuristic 0x100348244 / 0x100348530
  离开时 invGamma：pow(·, 2.2)，比例 0.36

removeCast 编排器 0x100373ea4
  atlas_matting
    propagation 0x10042e39c（闭式 α + 三维距离）
    randomWalk 0x10042e8cc
  reduceShadowMatting 0x1003bcbb4
  dilatePenumbraFast 0x1003bab5c
  getImageEdges 0x1003fab00
  比例反演 0x1003f68f4
  atlas inpaint 0x10043fa0c
    max(W,H) ≤ 1024 → inpaintCholesky → CHOLMOD 0x101b29e0c
    max(W,H) > 1024 → 缩小、递归、放大、inpaintCG
                      正则 1e-6，容差 1e-4，最多 5000 步
  suppressUnknownColors 0x1003da650（Lab 116/500/200）
  makeTeleportMap 0x1003feeac（无接缝则空表）
```

---

## 5. 地址对照

| 算子 | 地址 | 钉死的量 |
|---|---|---|
| 高光 sigmoid | `0x100348110` | \(\sigma(8\Delta V-2)\)，结果 \((w,w,w)\) |
| HSV 变换 | `0x1011c70cc` / `0x1011c6ee0` | 标准六扇区，\(V\) 在 `+8` |
| AO 亮度 | `0x100348530` | \(0.21,0.72,0.07\)，再与 \(\sigma(4-100\delta)\) 取最小 |
| 伽马 worker | `0x1009ae4b0`、`0x1009ae54c` | 乘 \(1/0.36\)，再 \(^{\,1/2.2}\) |
| 逆伽马 | `0x1009af12c` | 幂 \(2.2\) |
| 方向查询 | `0x10036faa0` | AO 在相邻金字塔层之间线性混合 |
| 比例夹紧 | `0x1003691b8` | \([0.05,20]\) |
| 法线引导滤波 | `0x1003fda00`、`0x1003fda78` | \(\epsilon=0.001\) 然后 \(0.05\)；半径 \(0.01M\) 然后 \(0.001M\) |
| matte 削减 | `0x1003bcde8` | \(m\cdot\operatorname{clamp}((m-\ell)/(h-\ell),0,1)\) |
| 抽边 | `0x1003fab00` | 阈值 \(0.2/0.1/0.05\)，概率 \(1/(2b^2)\) 等 |
| 光照比 | `0x1003f68f4` | \((B/(A+\epsilon)-(1-m))/(m+\epsilon)\)，夹到 \([0.05,20]\)，\(>1\) 丢弃 |
| Lab | `0x1003da650` | 116、500、200，分段立方根 |
| matte \(\alpha\) | `0x10042e648` | 三通道闭式投影，\(\epsilon=10^{-6}\) |
| 修补分流 | `0x10043fbf0` | `max(W,H)` 与 1024 比较 |
| CG 参数 | `0x100442d24`、`0x100442eb8` | \(10^{-6}\)，\(10^{-4}\)，5000 |
| 传送表 | `0x1003feeac`、`0x1003ff9e0` | 计数 \(<1\) 则空表；UV 排斥半径平方 25 |

---

## 6. 本轮没有钉死的部分

这些条目在调试字符串或前序笔记里有形状，本轮没有对应的指令窗口，不能当成已还原的公式：

- `LLAvg` 球面高斯的 \(\sigma\)，以及极点 \(1/\sin^2\theta\) 的具体加权。
- 主色统计里另一套亮度权（有笔记写成 \(0.21/0.71/0.08\)）。AO 路径用的是上一节的 \(0.21/0.72/0.07\)。
- `dilatePenumbraFast` 的距离模板，以及半影宽度换算成像素的除数。
- `getImageEdges` 外围的 Canny 阈值、盒式平滑半径。
- 高光抑制的分位点反推。
- Lab 距离返回之后如何乘回 \(m\)。
- 引导滤波内部的下采样档和 \(3\times 3\) 盒式迭代次数。
- 传送表三维近邻的 \(K\)，以及“第一个跨岛候选”的接受顺序。
- 弹簧构建时 4 邻域去重的完整循环。系数 \(\pm 1\) 与权 1 已有立即数和字符串，去重代码没有读完。
- `randomWalk` 的指数收缩步长，只核对到函数入口和共享的 \(10^{-6}\)。
