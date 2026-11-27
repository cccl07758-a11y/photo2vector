

# **Agisoft Texture De-Lighter管线解码：几何辅助固有图像分解技术蓝图**

## **1\. 摘要与系统架构推断**

对Agisoft Texture De-Lighter软件所依赖的许可证文件夹进行分析，揭示了其核心技术架构是一个高度复杂、结合了三维几何先验与高性能计算的系统，旨在解决固有图像分解（Intrinsic Image Decomposition, IID）这一欠定性难题。许可证目录中明确列出的关键库包括 guided-filter（引导滤波器）、poisson（泊松方程）、cholmod（Cholesky分解模块）、以及多个三维几何处理库如 draco、alembic 和 openmesh。

该系统的核心目标是自动将摄影测量生成的纹理（通常带有烘焙的光照和阴影）分离成纯粹的表面反照率（Albedo，即物体固有色）和光照贴图（Shading，即阴影和高光） 1。这项分析推断，Agisoft的去光照算法并非简单的图像滤波，而是一个混合计算模型，其架构主要分为两个紧密相连的阶段：

1. **初始光照估计与边缘保持：** 利用**引导滤波器** (guided-filter) 的结构转移特性，结合三维几何数据（如法线或环境光遮蔽图Ambient Occlusion, AO）作为指导图像，快速而精确地将光照和几何相关的阴影从纹理中分离出来 2。  
2. **全局优化与无缝重构：** 采用**梯度域优化**技术（即泊松方程）来强制实施全局光照一致性，并无缝集成用户提供的约束（如标记的“亮区”和“暗区”）3。这个大规模稀疏线性系统通过高性能的**CHOLMOD** (cholmod) 库进行求解，确保了计算效率和结果的稳健性 5。

这种多模块的组合表明，Agisoft的解决方案超越了传统的基于Retinex的去光照方法，而是建立在一个以精确三维几何为基础的，结合局部滤波和全局优化的计算摄影学框架之上。

## **2\. 理论框架：固有图像分解 (IID)**

固有图像分解是去光照任务的数学基础。它将观察到的图像强度视为两个独立分量的乘积，是逆向渲染（Inverse Rendering）中的一个基础性难题。

### **2.1 反射率-光照模型与欠定性挑战**

图像形成的物理模型定义了观察到的图像强度 $I$ 为表面固有颜色（反射率或反照率 $R$）与表面接收到的光照（光照度或阴影 $S$）的逐元素乘积 6：

$$I(x,y) \= R(x,y) \\odot S(x,y)$$为了将这个乘法问题转化为加法问题，更便于数学处理，分解过程通常在对数域中进行 7：$$\\log(I) \= \\log(R) \+ \\log(S) \\quad \\text{或表示为} \\quad i \= r \+ s$$

欠定性 (Ill-Posed Nature) 是IID的核心挑战。从一个输入函数 $i$ 中恢复两个未知函数 $r$ 和 $s$ 存在无数种可能的解 7。缺乏足够的约束或先验知识，会导致结果模糊且不准确，例如纹理细节可能会错误地泄漏到光照层中，或者光照信息错误地混入反照率层 6。因此，实现高保真分解依赖于引入强而准确的外部约束，即先验。

### **2.2 利用先验解决欠定性问题**

Agisoft的架构中整合了三种关键先验，这些先验的结合构成了其“先进算法”的基础 1。

| 先验类型 | 描述 | Agisoft系统中的来源 | 去光照中的作用 |
| :---- | :---- | :---- | :---- |
| **几何先验 (Shading Prior)** | 显式的三维数据，如表面法线、曲率和预计算的环境光遮蔽图（AO）8。 | draco, alembic, openmesh 数据处理 9。 | 提供客观、去纹理的光照估计 ($S\_{geo}$)，防止纹理细节泄漏到光照层中。 |
| **反射率先验 (Albedo Prior)** | 假设反照率在局部是分段恒定的，或其梯度是稀疏的 7。 | 泊松求解器实施的梯度域正则化和引导滤波器的结构转移特性 \[2, 6\]。 | 确保高频细节（纹理）被精确地隔离并保留在反照率 $R$ 中。 |
| **约束先验 (Boundary Conditions)** | 用户通过粗略笔触定义的已知光照区域（亮区/暗区）10。 | qt 图形用户界面 (GUI) 模块 10。 | 为全局泊松优化提供绝对的光照值约束（锚点），对稳定大规模线性系统至关重要。 |

通过系统性地结合这些先验，该软件将物理模型、用户指导和几何信息整合在一起，极大地约束了原本欠定的分解问题。

## **3\. 几何基础与数据处理**

Agisoft De-Lighter的设计初衷是处理摄影测量模型（如Metashape生成）的纹理 1。因此，对三维几何数据的精确处理是其独特之处，也是其鲁棒性的来源。

### **3.1 几何数据作为去光照的鲁棒先验**

在处理从现实世界采集的图像时，光照和阴影信息是与几何结构强相关的。光照变化往往发生在几何边缘或曲率变化大的地方 8。因此，准确的几何模型（即网格）是区分表面固有颜色变化和光照变化的关键 9。

Agisoft系统通过几何网格生成**几何先验 $S\_{geo}$**，例如光滑的法线贴图或AO图。这些数据代表了纯粹的、与纹理无关的光照基底，可用于指导后续的滤波和优化过程 9。例如，Unity Labs的去光照工具也强调了使用环境光遮蔽（AO）和法线贴图来移除烘焙光照的重要性，而Agisoft系统显然采取了类似甚至更复杂的策略。几何数据提供了超越单纯2D图像分析的稳健性，尤其是在处理具有复杂遮挡和自阴影的场景时 8。

### **3.2 模块分析：draco、alembic 和 openmesh 的作用**

许可证列表中出现的三个几何处理库，揭示了该软件在处理大型摄影测量模型时的工程需求和数据流程：

* **draco（压缩库）：** draco 库专注于压缩和解压缩三维几何网格和点云 12。考虑到摄影测量模型通常包含数百万个多边形和高分辨率纹理，高效地存储和传输几何数据至关重要，draco 在这里扮演了性能优化的角色。  
* **alembic（数据交换格式）：** alembic 是一个用于在不同三维应用程序间交换复杂场景数据的标准 13。它用于导入和管理网格拓扑、顶点数据和关键的UV坐标 14。这保证了从网格中提取的几何先验（如AO）可以精确地映射到纹理贴图的像素上。  
* **openmesh（网格处理）：** openmesh 提供了用于处理和修改网格的核心功能 15。在去光照流程中，该库可能用于查询顶点法线、计算曲率、或执行复杂的网格分析以生成高质量的 $S\_{geo}$。例如，网格的几何信息被用于估计更精确的光照度，特别是在深度突变或光照变化剧烈的区域 8。

这三个模块共同确认了Agisoft De-Lighter的管线始于对三维模型的高级几何预处理，而非仅依赖于2D图像像素。几何数据的准确性直接决定了光照分离的质量。

## **4\. 核心算法 I：引导滤波器的光照估计**

guided-filter 模块的纳入标志着系统使用了一种高效、保持边缘的局部滤波方法，用于对光照层进行初步估计和去噪。

### **4.1 引导滤波器 (GF) 的数学原理**

引导滤波器 16 是一种多功能图像处理工具，通过使用一张指导图像 $I$ 来滤波输入图像 $P$。其核心假设是，在以像素 $k$ 为中心的一个局部窗口 $\\omega\_k$ 内，滤波输出 $Q$ 是指导图像 $I$ 的线性变换 16：

$$Q\_j \= a\_k I\_j \+ b\_k, \\quad \\forall j \\in \\omega\_k$$

其中 $a\_k$ 和 $b\_k$ 是通过最小化一个包含正则化项的能量函数计算得出的局部系数。这种机制使得 GF 在去噪的同时，能够有效地保留指导图像中的边缘结构 2。

### **4.2 引导滤波在去光照中的应用：结构转移**

在Agisoft的IID管线中，引导滤波器被用于实现**结构转移 (Structure Transference)** 2。

* **输入与指导图像：** 对数域中的输入图像 $i$ 或其初步光照估计 $s\_{init}$ 被用作输入 $P$。从3D网格计算出的几何先验 $S\_{geo}$（例如光滑的AO图或法线光照）被用作指导图像 $I$ 18。  
* **分离过程：** GF 强制输出的光照估计 $s\_{est}$ 继承指导图像 $S\_{geo}$ 中定义的结构。由于 $S\_{geo}$ 是从光滑的几何中提取的，它包含清晰的阴影边界和几何形状，但不包含高频的表面纹理 9。因此，通过引导滤波器的操作，任何与几何结构不符的高频变化（即纹理）都会被有效地从 $s\_{est}$ 中分离出来，从而保留在计算出的反照率残差 $r\_{est} \= i \- s\_{est}$ 中。  
* 这种方法是线性的，计算复杂度低 ($O(N)$)，因此能够提供快速且高质量的初步光照分离结果，为后续的全局优化奠定基础 17。

## **5\. 核心算法 II：梯度域优化与稀疏线性系统求解**

虽然引导滤波器擅长局部结构保持，但它无法确保结果的全局一致性，也难以直接融合用户提供的绝对约束。因此，系统必须采用全局优化方法，即泊松方程求解。

### **5.1 基于泊松方程的全局优化**

泊松方程广泛应用于计算机图形学和图像处理的梯度域操作中，用于实现无缝的图像合成、编辑和求解固有分解问题 4。在去光照中，其目的是找到一个最终的光照图 $s^\*$，使其 Laplacian ( $\\nabla^2 s^\*$ ) 与预先设计的目标梯度场 $\\text{div}(\\mathbf{v})$ 的散度相匹配 3：

$$\\nabla^2 s^\* \= \\text{div}(\\mathbf{v})$$

目标梯度场 $\\mathbf{v}$ 的设计至关重要。它通常结合了引导滤波器生成的初步光照 $s\_{est}$ 的梯度，并施加正则化，以确保反照率 $r$ 的梯度稀疏（即反照率变化小），从而将纹理稳定地保持在 $R$ 分量中 6。通过最小化一个综合了数据项和正则化项的能量泛函 19，泊松求解确保了最终的光照图 $s^\*$ 在全局上光滑且一致，并能无缝地融入用户定义的约束（如亮区/暗区笔触）。

### **5.2 CHOLMOD：高分辨率稀疏线性系统的求解**

对二维图像上的泊松微分方程进行离散化，通常采用五点模板的有限差分法 20，这会产生一个大规模的线性方程组：

$$A \\mathbf{x} \= \\mathbf{b}$$

其中 $N$ 是像素总数，$A$ 是一个 $N \\times N$ 的稀疏矩阵（表示离散化的拉普拉斯算子），$\\mathbf{x}$ 是待求的 log 光照值向量（ $s^\*$ ），而 $\\mathbf{b}$ 则包含目标梯度场的散度以及用户约束的边界值 4。  
解决这个大规模稀疏系统是计算瓶颈。cholmod (CHOLesky MODification) 是 SuiteSparse 线性代数包的一部分，专门用于对稀疏对称正定矩阵进行高性能 Cholesky 分解 5。Agisoft选择 **CHOLMOD** 这一**直接求解器**而非迭代求解器（如共轭梯度法），表明了对解的**鲁棒性和精确性**的严格要求。直接求解器在矩阵条件数较差或需要绝对精度时表现更可靠 5。

此外，CHOLMOD 自 4.0.0 版本以来支持 GPU 加速，对于稀疏分解操作，性能可提升 $3\\times$ 或更高 5。这种对高性能稀疏计算的依赖，是将高保真、全局优化的 IID 算法应用于高分辨率摄影测量纹理所需的工程优化。

以下表格对比了引导滤波器和泊松优化在去光照管线中的互补作用：

互补算法对比：引导滤波器与泊松优化

| 特征 | 引导滤波器（滤波方法） | 泊松方程优化（优化方法） |
| :---- | :---- | :---- |
| **基础数学原理** | 局部线性回归与结构转移 16。 | 全局变分能量最小化，通过拉普拉斯算子实现 \[19, 22\]。 |
| **计算复杂度** | 高效，线性复杂度 $O(N)$ 17。 | 较高，通过稀疏直接分解求解，复杂度通常高于 $O(N \\log N)$，依赖 cholmod 加速 5。 |
| **主要功能** | 快速、局部结构保持，初步分离光照与纹理。 | 全局一致性，无缝融合约束，消除低频光照不一致性 4。 |
| **指导信息** | 几何先验（AO/法线）2。 | 目标梯度场 $\\mathbf{v}$ 和用户提供的绝对约束 3。 |

## **6\. Agisoft De-Lighter 完整管线流程合成**

将分析出的模块和算法结合，可以构建出Agisoft De-Lighter的完整工作流程，这是一个精心设计的多阶段计算管线：

1. **3D 数据导入与几何预处理：**  
   * 通过 alembic 导入网格模型和纹理，并使用 draco 进行高效数据处理 12。  
   * openmesh 用于处理网格几何，计算和提取高精度的几何先验 $S\_{geo}$（如环境光遮蔽图和光滑法线光照图）8。  
2. **对数域转换与初始估计：**  
   * 将输入纹理 $I$ 转换到对数域 $i$ 7。  
   * 可能有初步的光照估计 $s\_{init}$，可能利用 opencv 等库中的模糊技术（例如高斯模糊或双边滤波）实现多尺度 Retinex 模型的初始估计 23。  
3. **光照细化（引导滤波阶段）：**  
   * 利用 $S\_{geo}$ 作为指导图像，对 $i$ 或 $s\_{init}$ 应用引导滤波器，生成保持几何边缘的精炼光照估计 $s\_{est}$ 2。  
   * 计算初步反照率残差 $r\_{est} \= i \- s\_{est}$。  
4. **用户约束的引入：**  
   * 用户通过 qt 界面提供的粗略笔触（标记的亮区和暗区）被转化为 $A\\mathbf{x} \= \\mathbf{b}$ 系统中的硬性数据约束 $C$ 10。  
5. **全局优化（泊松/CHOLMOD阶段）：**  
   * 构建目标梯度场 $\\mathbf{v}$，该梯度场旨在保留 $r\_{est}$ 中的高频细节，并同时保持 $s\_{est}$ 的几何结构和全局光滑性 3。  
   * 组装包含梯度信息和约束 $C$ 的稀疏线性系统 $A\\mathbf{x} \= \\mathbf{b}$。  
   * 利用 **CHOLMOD** 对稀疏矩阵 $A$ 进行 Cholesky 分解并求解，得到最终全局优化后的 log 光照 $s^\*$ 5。  
6. **最终重构：**  
   * 计算最终反照率 $r^\* \= i \- s^\*$。  
   * 对 $r^\*$ 和 $s^\*$ 进行指数化处理，获得最终的去光照纹理 $R^\*$ 和分离的光照贴图 $S^\*$。

## **7\. 自定义去光照工具的开发蓝图**

针对希望自行制作图片去光照工具的用户，成功的实现需要根据目标质量和可用资源选择适当的策略。

### **7.1 策略一：基于滤波器的快速实现**

这种策略旨在快速实现局部去光照效果，但其质量高度依赖于输入纹理的特性或几何先验的获取。

* **技术路线：** 采用基于 Retinex 的方法或引导滤波。Retinex 算法将图像分解为反射率（即去光照结果）和光照度，通过对光照度进行平滑估计实现分离 25。  
* **实现要点：** 可以使用 opencv 库实现多尺度 Retinex (MSR) 24。通过应用高斯模糊或双边滤波 (cv.bilateralFilter()) 对对数域的输入图像进行平滑处理，估算出光照层 23。如果能获得几何信息（如法线或深度图），则应实现引导滤波器 16，将几何信息作为指导图像 $I$ 以提高边缘保持能力 2。  
* **局限性：** 缺乏全局约束，结果可能存在低频的光照不一致或“光晕”现象，且难以精确融合用户指定的绝对光照区域。

### **7.2 策略二：全局优化与稀疏求解器集成**

要达到Agisoft的商业级去光照效果，必须采用全局优化方法。

* **技术路线：** 实现基于泊松方程的梯度域优化，这是唯一能够实现全局无缝重构和精确约束集成的途径 4。  
* **实现步骤：**  
  1. **梯度场设计：** 仔细设计能量泛函和目标梯度场 $\\mathbf{v}$，确保其在纹理区域服从反照率稀疏先验 3。  
  2. **稀疏矩阵构建：** 将泊松方程离散化，构造出大规模的稀疏拉普拉斯矩阵 $A$ 20。  
  3. **约束集成：** 将用户标记的亮/暗区（$C$）转化为方程组 $\\mathbf{b}$ 中的绝对值约束。  
  4. **求解器集成：** 集成一个高性能的稀疏线性求解器。在C++环境中，直接使用 CHOLMOD 5 是最稳健的选择。在Python环境中，可以使用 scikit-sparse 等包装库间接调用 CHOLMOD 的功能 21。由于图像尺寸巨大，对 $A\\mathbf{x} \= \\mathbf{b}$ 进行高性能求解是实现商业可行性的关键挑战。

结论是，Agisoft Texture De-Lighter通过融合三维几何先验、高效的引导滤波、以及高性能稀疏求解器驱动的泊松全局优化，解决了固有图像分解中的欠定性问题。用户若想复制其功能，则必须克服在百万像素级别上构建和求解稀疏线性系统的计算障碍，这是实现高质量去光照效果的必要条件。

#### **Works cited**

1. Agisoft Texture De-Lighter: General Workflow and Best Practices, accessed November 6, 2025, [https://www.agisoftmetashape.com/agisoft-texture-de-lighter-general-workflow-and-best-practices/](https://www.agisoftmetashape.com/agisoft-texture-de-lighter-general-workflow-and-best-practices/)  
2. What is Guided Image Filtering? \- MATLAB & Simulink \- MathWorks, accessed November 6, 2025, [https://www.mathworks.com/help/images/what-is-guided-image-filtering.html](https://www.mathworks.com/help/images/what-is-guided-image-filtering.html)  
3. Single Image Intrinsic Decomposition Without a Single Intrinsic Image, accessed November 6, 2025, [https://www.ecva.net/papers/eccv\_2018/papers\_ECCV/papers/Wei-Chiu\_Single\_Image\_Intrinsic\_ECCV\_2018\_paper.pdf](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Wei-Chiu_Single_Image_Intrinsic_ECCV_2018_paper.pdf)  
4. The Poisson Equation in Image & Shape Processing (600.657), accessed November 6, 2025, [https://www.cs.jhu.edu/\~misha/Fall07/](https://www.cs.jhu.edu/~misha/Fall07/)  
5. CHOLMOD | NVIDIA Developer, accessed November 6, 2025, [https://developer.nvidia.com/cholmod](https://developer.nvidia.com/cholmod)  
6. Texture-aware Intrinsic Image Decomposition with Model- and Learning-based Priors \- arXiv, accessed November 6, 2025, [https://arxiv.org/pdf/2509.09352?](https://arxiv.org/pdf/2509.09352)  
7. Intrinsic Image Decomposition Using a Sparse Representation of Reflectance, accessed November 6, 2025, [https://www.researchgate.net/publication/258055385\_Intrinsic\_Image\_Decomposition\_Using\_a\_Sparse\_Representation\_of\_Reflectance](https://www.researchgate.net/publication/258055385_Intrinsic_Image_Decomposition_Using_a_Sparse_Representation_of_Reflectance)  
8. Intrinsic Image Decomposition Using Point Cloud Representation \- arXiv, accessed November 6, 2025, [https://arxiv.org/html/2307.10924v2](https://arxiv.org/html/2307.10924v2)  
9. Download Unity Labs' free texture de-lighting tool | CG Channel, accessed November 6, 2025, [https://www.cgchannel.com/2017/07/download-unity-labs-free-texture-de-lighting-tool/](https://www.cgchannel.com/2017/07/download-unity-labs-free-texture-de-lighting-tool/)  
10. Agisoft Texture De-Lighter \- general workflow \- Helpdesk Portal, accessed November 6, 2025, [https://agisoft.freshdesk.com/support/solutions/articles/31000158376-agisoft-texture-de-lighter-general-workflow](https://agisoft.freshdesk.com/support/solutions/articles/31000158376-agisoft-texture-de-lighter-general-workflow)  
11. Setting Up Textures & Lighting for a Realistic 3D Scene \- 80 Level, accessed November 6, 2025, [https://80.lv/articles/001agt-setting-up-textures-lighting-for-a-realistic-3d-scene](https://80.lv/articles/001agt-setting-up-textures-lighting-for-a-realistic-3d-scene)  
12. Draco 3D Graphics Compression \- Google, accessed November 6, 2025, [https://google.github.io/draco/](https://google.github.io/draco/)  
13. Alembic geometry node \- SideFX, accessed November 6, 2025, [https://www.sidefx.com/docs/houdini/nodes/sop/alembic.html](https://www.sidefx.com/docs/houdini/nodes/sop/alembic.html)  
14. Maya Help | Modifying an Alembic Procedural Using Operators | Autodesk, accessed November 6, 2025, [https://help.autodesk.com/view/MAYAUL/2025/ENU/?guid=arnold\_for\_maya\_operators\_am\_Modifying\_an\_Alembic\_Procedural\_Using\_Operators\_html](https://help.autodesk.com/view/MAYAUL/2025/ENU/?guid=arnold_for_maya_operators_am_Modifying_an_Alembic_Procedural_Using_Operators_html)  
15. Deleting geometry elements \- OpenMesh \- RWTH Aachen, accessed November 6, 2025, [https://www.graphics.rwth-aachen.de/media/openmesh\_static/Documentations/OpenMesh-6.0-Documentation/a00060.html](https://www.graphics.rwth-aachen.de/media/openmesh_static/Documentations/OpenMesh-6.0-Documentation/a00060.html)  
16. Guided Filter-Inspired Network for Low-Light RAW Image Enhancement \- MDPI, accessed November 6, 2025, [https://www.mdpi.com/1424-8220/25/9/2637](https://www.mdpi.com/1424-8220/25/9/2637)  
17. Guided Image Filtering for Interactive High-quality Global Illumination | Request PDF, accessed November 6, 2025, [https://www.researchgate.net/publication/220506983\_Guided\_Image\_Filtering\_for\_Interactive\_High-quality\_Global\_Illumination](https://www.researchgate.net/publication/220506983_Guided_Image_Filtering_for_Interactive_High-quality_Global_Illumination)  
18. Reflectance Adaptive Filtering Improves Intrinsic Image Estimation \- GitHub, accessed November 6, 2025, [https://github.com/tnestmeyer/reflectance-filtering](https://github.com/tnestmeyer/reflectance-filtering)  
19. A Simple Model for Intrinsic Image Decomposition with Depth Cues \- ResearchGate, accessed November 6, 2025, [https://www.researchgate.net/publication/256663553\_A\_Simple\_Model\_for\_Intrinsic\_Image\_Decomposition\_with\_Depth\_Cues](https://www.researchgate.net/publication/256663553_A_Simple_Model_for_Intrinsic_Image_Decomposition_with_Depth_Cues)  
20. ALAFF Where do sparse linear systems come from? \- Texas Computer Science, accessed November 6, 2025, [https://www.cs.utexas.edu/\~flame/laff/alaff/chapter07-why-sparse-2d.html](https://www.cs.utexas.edu/~flame/laff/alaff/chapter07-why-sparse-2d.html)  
21. Sparse Cholesky decomposition (sksparse.cholmod) \- Read the Docs, accessed November 6, 2025, [https://scikit-sparse.readthedocs.io/en/latest/cholmod.html](https://scikit-sparse.readthedocs.io/en/latest/cholmod.html)  
22. Smoothing Images \- OpenCV Documentation, accessed November 6, 2025, [https://docs.opencv.org/4.x/d4/d13/tutorial\_py\_filtering.html](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)  
23. Low-Illumination Image Enhancement Algorithm Based on Improved Multi-Scale Retinex and ABC Algorithm Optimization \- PMC \- PubMed Central, accessed November 6, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9035903/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9035903/)  
24. Image Enhancement using Retinex Algorithms \- Santha Lakshmi Narayana, accessed November 6, 2025, [https://santhalakshminarayana.github.io/blog/retinex-image-enhancement](https://santhalakshminarayana.github.io/blog/retinex-image-enhancement)  
25. Enhancing the Quality of Low-Light Printed Circuit Board Images through Hue, Saturation, and Value Channel Processing and Improved Multi-Scale Retinex \- Scirp.org., accessed November 6, 2025, [https://www.scirp.org/journal/paperinformation?paperid=130349](https://www.scirp.org/journal/paperinformation?paperid=130349)