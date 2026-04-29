<!--Copyright © 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# NVIDIA GPU架构

## 学习目标

1. 了解NVIDIA GPU从Fermi到Blackwell九代架构的发展历程
2. 掌握NVIDIA各代架构的关键创新和技术特点
3. 深入理解Tensor Core的演进和各代架构的性能提升
4. 掌握CUDA编程模型及其与硬件架构的对应关系
5. 理解NVLink与NVSwitch的技术演进和带宽发展

## 章节导言

NVIDIA是全球领先的GPU制造商，其GPU架构演进代表了GPU技术的发展方向。从2010年Fermi架构到2024年Blackwell架构，NVIDIA GPU经历了9代重大演进，在AI计算领域建立了难以撼动的技术领先地位。

本章将系统梳理NVIDIA GPU架构的发展脉络，深入分析每一代架构的关键技术创新。从Fermi确立GPU计算基础架构，到Pascal引入NVLink互联技术，再到Volta开创Tensor Core时代，NVIDIA不断推动GPU计算能力的边界。

本章还将详细介绍CUDA编程模型与硬件架构的对应关系，帮助读者理解软件如何映射到硬件执行。同时，NVLink和NVSwitch作为多GPU互联的关键技术，其演进历程也是理解现代AI计算集群的重要基础。

通过本章学习，读者将对NVIDIA GPU架构有全面深入的认识，理解其技术领先的原因，并为实际开发工作提供理论指导。

## 3.1 NVIDIA GPU架构演进概述

### 3.1.1 九代架构一览

NVIDIA GPU架构经历了9代重要演进：

| 架构名称 | 中文名 | 发布年份 | 核心技术突破 |
|---------|--------|---------|-------------|
| Fermi | 费米 | 2010 | 完整GPU计算架构 |
| Kepler | 开普勒 | 2012 | 多核并行优化 |
| Maxwell | 麦克斯韦 | 2014 | 能效大幅提升 |
| Pascal | 帕斯卡 | 2016 | NVLink、HBM2 |
| Volta | 伏特 | 2017 | Tensor Core |
| Turing | 图灵 | 2018 | RT Core、Tensor Core 2.0 |
| Ampere | 安培 | 2020 | Tensor Core 3.0、MIG |
| Hopper | 赫柏 | 2022 | 异构CPU+GPU、HBM3 |
| Blackwell | 布莱克韦尔 | 2024 | 第二代Transformer Engine |

### 3.1.2 命名渊源

NVIDIA GPU架构以著名科学家命名：

- **费米（Enrico Fermi）**：意大利物理学家，量子力学和统计力学先驱
- **开普勒（Johannes Kepler）**：德国天文学家，行星运动定律
- **麦克斯韦（James Clerk Maxwell）**：电磁理论之父
- **帕斯卡（Blaise Pascal）**：法国数学家，概率论奠基人
- **伏特（Alessandro Volta）**：意大利物理学家，电池之父
- **图灵（Alan Turing）**：计算机科学之父
- **安培（André-Marie Ampère）**：电磁学之父
- **赫柏（Grace Hopper）**：计算机编程先驱
- **布莱克韦尔（David Blackwell）**：统计学家

## 3.2 Fermi架构（2010）

### 3.2.1 Fermi架构概述

Fermi是NVIDIA第一个完整的GPU计算架构，标志着GPU从图形处理向通用计算的全面转型。

**核心参数**：

- 16个SM（流式多处理器）
- 每个SM 32个CUDA Core
- 共512个CUDA Core
- 6个64位内存控制器（共384位）
- 最大6GB GDDR5显存
- 支持ECC内存
- 40亿晶体管（40nm/28nm）

### 3.2.2 Fermi架构创新

**完整的GPU计算架构**

Fermi首次真正从硬件层面支持GPU计算：

- **CUDA Core**：完整的IEEE 754-2008浮点运算单元
- **双精度支持**：满足科学计算需求
- **L1/L2缓存**：首次在GPU上支持缓存层次
- **ECC支持**：提高计算可靠性
- **统一虚拟地址空间**：简化编程模型

**流式多处理器（SM）结构**

每个SM包含：

- 32个CUDA Core（FP32）
- 16个LD/ST单元
- 4个SFU（特殊函数单元）
- 1个Warp调度器
- 64KB共享内存/L1缓存

### 3.2.3 Fermi的局限性

Fermi架构也存在一些局限：

- 功耗较高
- 共享内存和L1缓存竞争
- 双精度性能相对较弱
- 架构仍在图形和计算之间平衡

## 3.3 Kepler架构（2012）

### 3.3.1 Kepler架构概述

Kepler架构在Fermi基础上大幅提升能效，专为Tesla计算卡设计。

**核心参数**：

- 15个SMX
- 每个SMX 192个FP32 + 64个FP64 CUDA Core
- 共2880个FP32 Core
- GPU Boost技术
- 动态并行支持
- 71亿晶体管（28nm）

### 3.3.2 Kepler关键创新

**SMX设计**

Kepler将SM升级为SMX，一个SMX包含：

- 192个FP32 CUDA Core
- 64个FP64 CUDA Core
- 32个LD/ST单元
- 32个SFU
- 4个Warp调度器

**能效提升**

相比Fermi：

- 每瓦性能提升3倍
- 通过减少控制单元比例，增加计算单元
- GPU Boost动态超频

**新技术支持**

- **动态并行**：GPU可自行启动新内核
- **Hyper-Q**：多个CPU核心同时使用GPU
- **Grid Management Unit**：管理网格调度

## 3.4 Maxwell架构（2014）

### 3.4.1 Maxwell架构概述

Maxwell是NVIDIA历史上能效提升最显著的架构之一。

**核心参数**：

- 16个SMM
- 每个SMM 128个CUDA Core
- 共2048个CUDA Core
- 动态负载均衡
- 80亿晶体管（28nm）

### 3.4.2 Maxwell技术特点

**SMM（Maxwell流式多处理器）**

每个SMM包含：

- 4个逻辑处理块
- 每块32个CUDA Core
- 8个LD/ST + 8个SFU
- 独立Warp调度器

**能效优化策略**

- 减少每组Core数量，降低复杂度
- 增加每SM的线程数量
- 更大的共享内存（64KB vs 48KB）
- 更大的L2缓存（2MB vs 768KB）

**与Kepler对比**：

| 指标 | Kepler GK104 | Maxwell GM204 |
|------|-------------|---------------|
| CUDA Core | 1536 | 2048 |
| 核心频率 | 1006 MHz | 1126 MHz |
| 浮点性能 | 3090 GFLOPS | 4612 GFLOPS |
| 共享内存/SM | 48KB | 96KB |
| L2缓存 | 512KB | 2048KB |
| TDP | 195W | 165W |

## 3.5 Pascal架构（2016）

### 3.5.1 Pascal架构概述

Pascal是NVIDIA历史上最具创新性的架构之一，引入了多项革命性技术。

**核心参数**：

- 60个SM
- 每个SM 64个FP32 + 32个FP64 CUDA Core
- 16nm FinFET工艺
- HBM2显存
- NVLink高速互联
- 153亿晶体管

### 3.5.2 Pascal关键创新

**HBM2显存**

Pascal首次在GPU中使用HBM2：

- 带宽：720 GB/s（vs 336 GB/s）
- 效率：每比特能耗降低4倍
- 面积：GPU + 封装集成

**NVLink 1.0**

革命性的GPU互联技术：

- 双向带宽：160 GB/s
- 是PCIe 3.0 x16的5倍
- 支持GPU直接内存访问

**统一内存**

- GPU和CPU共享虚拟地址空间
- 支持内存页面迁移
- 简化异构编程

### 3.5.3 Pascal的SM设计

Pascal的SM进行了精简优化：

- 每SM 64个FP32 Core（分成两组各32）
- 每SM 32个FP64 Core
- 更大寄存器文件
- 支持更多并发Block

## 3.6 Volta架构（2017）

### 3.6.1 Volta架构概述

Volta是NVIDIA第一个专门为深度学习设计的架构，引入了革命性的Tensor Core。

**核心参数**：

- 80个SM
- 每个SM 64个FP32 + 64个INT32 + 8个Tensor Core
- 12nm FFN工艺
- HBM2显存
- NVLink 2.0
- NVSwitch 1.0
- 211亿晶体管

### 3.6.2 Volta关键创新

**Tensor Core 1.0**

革命性深度学习加速单元：

- 4×4矩阵运算单元
- 混合精度：FP16输入，FP32累加
- 每SM 8个Tensor Core
- 每周期：1024 FMA（2048 FLOPS）

**独立线程调度**

改进SIMT模型：

- 每个线程独立PC和Stack
- 细粒度同步和协作
- 解决Warp内分支效率问题

**FP32/INT32并发执行**

Volta可同时执行FP32和INT32：

- 深度学习：FP32用于计算
- 图形处理：INT32用于索引
- 效率提升

### 3.6.3 Volta SM结构

```
每个SM包含：
├── 64个FP32 Core
├── 64个INT32 Core
├── 32个FP64 Core
├── 8个Tensor Core
├── 32个LD/ST Unit
├── 16个SFU
├── 4个Warp调度器
├── 256 KB 寄存器文件
└── 96 KB 共享内存/L1
```

## 3.7 Turing架构（2018）

### 3.7.1 Turing架构概述

Turing是NVIDIA第一个支持光线追踪的消费级GPU架构。

**核心参数**：

- 72个SM
- 每SM 64个FP32 + 64个INT32 + 8个Tensor Core
- RT Core光线追踪单元
- 16nm TSMC工艺
- GDDR6显存
- 186亿晶体管

### 3.7.2 Turing关键创新

**RT Core 1.0**

专为光线追踪设计的硬件单元：

- 加速光线-三角形求交
- BVH边界体层次遍历
- 与传统光栅化混合使用

**Tensor Core 2.0**

新增INT8/INT4/Binary精度支持：

- INT8：256 OPS/Clock
- INT4：512 OPS/Clock
- 加速推理推理

**GDDR6显存**

相比GDDR5X：

- 14 Gbps传输速率
- 20%能效提升
- 更大容量支持

### 3.7.3 Turing RT Core原理

**光线追踪原理**

1. 光线从摄像机射出
2. 光线与场景物体求交
3. 计算交点处 shading
4. 递归处理反射、折射

**RT Core加速**

传统GPU使用ALU计算光线求交，RT Core使用专用硬件：

```
RT Core执行：
1. 测试包围盒（Box Test）
2. 遍历BVH层次结构
3. 光线-三角形求交
4. 返回交点信息
```

## 3.8 Ampere架构（2020）

### 3.8.1 Ampere架构概述

Ampere是NVIDIA数据中心的旗舰架构，在AI和HPC领域实现突破性性能。

**核心参数**：

- 108个SM
- 每SM 64个FP32 + 64个INT32 + 4个Tensor Core
- 7nm TSMC工艺
- HBM2e显存
- NVLink 3.0
- MIG多实例GPU
- 542亿晶体管

### 3.8.2 Ampere关键创新

**Tensor Core 3.0**

全新第三代Tensor Core：

- 新增TF32、BF16精度
- 结构稀疏性支持
- 每SM 4个Tensor Core

**TF32精度**

专为AI设计的新精度：

| 精度 | 指数位 | 尾数位 | AI适用性 |
|------|--------|--------|----------|
| FP32 | 8 | 23 | 基准 |
| FP16 | 5 | 10 | 好（动态范围不足） |
| BF16 | 8 | 7 | 优秀 |
| TF32 | 8 | 10 | 最优 |

TF32保持FP32的指数位和FP16的尾数位，兼顾动态范围和精度。

**多实例GPU（MIG）**

将单个A100划分为多个独立GPU：

- 最多7个实例
- 每个实例独立内存和计算资源
- 适用于云服务多用户场景

**结构稀疏性**

2:4结构稀疏支持：

- 训练后剪枝
- 压缩稀疏矩阵
- 2倍加速

### 3.8.3 Ampere SM结构

```
每个SM包含：
├── 64个FP32 Core
├── 64个INT32 Core
├── 32个FP64 Core
├── 4个Tensor Core
├── 32个LD/ST Unit
├── 16个SFU
├── 4个Warp调度器
├── 192 KB L1/共享内存
└── 256 KB 寄存器文件
```

**A100 vs V100性能对比**：

| 指标 | V100 | A100 | 提升 |
|------|------|------|------|
| FP32 | 15.7 TFLOPS | 19.5 TFLOPS | 1.2x |
| FP16 | 125 TFLOPS | 312 TFLOPS | 2.5x |
| BF16 | - | 312 TFLOPS | - |
| TF32 | - | 156 TFLOPS | - |
| INT8 | - | 1248 TOPS | - |
| 显存带宽 | 900 GB/s | 2039 GB/s | 2.3x |
| HBM容量 | 32 GB | 80 GB | 2.5x |

## 3.9 Hopper架构（2022）

### 3.9.1 Hopper架构概述

Hopper是NVIDIA第一个真正的异构计算平台，集成CPU和GPU。

**核心参数**：

- 144个SM
- 4nm TSMC工艺
- HBM3显存
- NVLink 4.0
- NVLink-C2C CPU-GPU互联
- 800亿晶体管

### 3.9.2 Hopper关键创新

**Grace CPU集成**

NVIDIA首次集成CPU：

- 72核ARM Neoverse V2
- 512GB LPDDR5X
- 900 GB/s NVLink-C2C带宽
- 1+1>2的协同效应

**Tensor Core 4.0**

第四代Tensor Core：

- 新增FP8精度
- Transformer Engine 2.0
- 动态范围自适应

**FP8精度**

专为Transformer设计：

- 1.25倍加速（vs FP16）
- 保持模型精度
- 特别适合LLM训练

**Transformer Engine 2.0**

专门优化Transformer：

- 自动精度选择
- 动态Tiling
- 重计算策略

### 3.9.3 NVLink 4.0

第四代NVLink：

- 900 GB/s双向带宽
- 18链路/GPU
- 支持256 GPU互联

### 3.9.4 H100规格

**单芯片H100**：

| 规格 | H100 SXM |
|------|----------|
| SM | 132 (实际使用) |
| CUDA Core | 16896 |
| Tensor Core | 528 |
| FP32 | 67 TFLOPS |
| FP64 | 34 TFLOPS |
| FP16 | 1981 TFLOPS |
| FP8 | 3959 TFLOPS |
| HBM3容量 | 80 GB |
| HBM3带宽 | 3.35 TB/s |
| TDP | 700W |

**DGX H100系统**：

- 8×H100 SXM5
- 1.6 TB HBM3
- 3.2 TB/s聚合带宽

## 3.10 Blackwell架构（2024）

### 3.10.1 Blackwell架构概述

Blackwell是NVIDIA最新的数据中心GPU架构，为生成式AI和大模型时代设计。

**核心参数**：

- 2080亿晶体管
- 4NP工艺（定制TSMC）
- 第二代Transformer Engine
- NVLink 5.0
- RAS引擎
- 安全AI

### 3.10.2 Blackwell关键创新

**双芯片设计**

突破光刻极限：

- 两个GPU die通过10 TB/s互联
- 统一内存寻址
- 接近单芯片性能

**第二代Transformer Engine**

- FP4精度支持
- 动态范围增强
- 更优的大模型训练

**NVLink 5.0**

- 1.8 TB/s双向带宽
- 576 GPU互联
- 支持万亿参数模型

**GB200 NVL72**

- 72 GPU + 36 CPU
- 液冷机架设计
- 1.44 EFLOPS FP4

### 3.10.3 性能对比

**B200 vs H100**：

| 指标 | H100 | B200 | 提升 |
|------|------|------|------|
| FP4 Tensor | - | 18 EFLOPS | - |
| FP8 Tensor | 3959 TFLOPS | 9000 TFLOPS | 2.3x |
| FP16/BF16 | 1981 TFLOPS | 4500 TFLOPS | 2.3x |
| 显存带宽 | 3.35 TB/s | 8 TB/s | 2.4x |
| GPU互联 | 900 GB/s | 1.8 TB/s | 2x |

**AI计算性能演进**：

2016年Pascal P100 → 2024年Blackwell B200：
8年提升1000倍！

## 3.11 CUDA编程详解

### 3.11.1 CUDA编程模型

CUDA是NVIDIA的并行计算平台和编程模型。

**基本概念**：

- **Host**：CPU及其内存
- **Device**：GPU及其内存
- **Kernel**：在GPU上执行的并行函数
- **Grid**：整个内核执行的所有线程
- **Block**：协作线程组
- **Thread**：最小执行单元

### 3.11.2 CUDA内存模型

```
┌─────────────────────────────────────┐
│         Global Memory (GPU)          │
│              ~80 GB                  │
├─────────────────────────────────────┤
│            L2 Cache (40 MB)          │
├─────────────────────────────────────┤
│      L1 Cache + Shared Memory        │
│            (192 KB/SM)                │
├─────────────────────────────────────┤
│           Register File              │
│            (256 KB/SM)               │
└─────────────────────────────────────┘
```

**内存类型**：

| 内存 | 位置 | 访问 | 作用域 |
|------|------|------|--------|
| 寄存器 | SM | R/W | 线程私有 |
| 本地内存 | Global | R/W | 线程私有 |
| 共享内存 | SM | R/W | Block内 |
| 全局内存 | Device | R/W | 全局 |
| 常量内存 | Global | R | 全局 |
| 纹理内存 | Global | R | 全局 |

### 3.11.3 CUDA编程示例

**向量加法**：

```c
// CUDA内核：向量加法
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Host代码
int main() {
    int N = 1 << 20;  // 100万元素
    
    // 分配GPU内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动内核
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    
    // 拷贝结果回CPU
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
```

**矩阵乘法**：

```c
__global__ void matrixMul(float *A, float *B, float *C, int N) {
    // 使用共享内存优化
    __shared__ float sA[16][16];
    __shared__ float sB[16][16];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int k = 0; k < N; k += 16) {
        // 协作加载到共享内存
        sA[threadIdx.y][threadIdx.x] = A[row * N + k + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        __syncthreads();
        
        // 计算
        for (int i = 0; i < 16; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

### 3.11.4 CUDA线程层次

```
Grid
│
├── Block (0,0)
│   ├── Warp 0: Thread 0-31
│   ├── Warp 1: Thread 32-63
│   └── ...
│
├── Block (1,0)
│   ├── Warp 0: Thread 0-31
│   └── ...
│
└── Block (m,n)
    └── ...
```

**线程束（Warp）**：

- 32个线程为一组
- 同一Warp执行相同指令
- 分支分化会降低效率

### 3.11.5 CUDA内存管理

**统一内存（Unified Memory）**：

```c
// 分配统一内存
float *data;
cudaMallocManaged(&data, size);

// 访问无需拷贝
// GPU和CPU自动迁移数据
data[0] = 1.0f;
cudaDeviceSynchronize();

// 自动回收
cudaFree(data);
```

**内存拷贝类型**：

```c
cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);    // CPU→CPU
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);  // CPU→GPU
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);  // GPU→CPU
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice); // GPU→GPU
```

## 3.12 NVLink与NVSwitch

### 3.12.1 NVLink发展历程

NVLink是NVIDIA开发的高速GPU互联技术。

**NVLink 1.0 (Pascal)**
- 2014年发布
- 双向带宽：80 GB/s/链路
- 4链路/GPU
- 320 GB/s总带宽

**NVLink 2.0 (Volta)**
- 2017年发布
- 双向带宽：100 GB/s/链路
- 6链路/GPU
- 600 GB/s总带宽

**NVLink 3.0 (Ampere)**
- 2020年发布
- 双向带宽：150 GB/s/链路
- 12链路/GPU
- 1.8 TB/s总带宽

**NVLink 4.0 (Hopper)**
- 2022年发布
- 双向带宽：200 GB/s/链路
- 18链路/GPU
- 3.6 TB/s总带宽

**NVLink 5.0 (Blackwell)**
- 2024年发布
- 双向带宽：200 GB/s/链路
- 18链路/GPU
- 1.8 TB/s双向带宽/GPU

### 3.12.2 NVSwitch

NVSwitch是NVLink交换系统，实现多GPU全互联。

**功能**：

- GPU间高速点对点通信
- 全互联拓扑
- GPU共享内存寻址

**历代NVSwitch**：

| 代数 | 支持GPU数 | 带宽 | 引入 |
|------|-----------|------|------|
| NVSwitch 1.0 | 16 | 2.4 TB/s | Volta |
| NVSwitch 2.0 | 16 | 4.8 TB/s | Ampere |
| NVSwitch 3.0 | 16 | 7.2 TB/s | Hopper |
| NVLink Switch | 576 | 1 PB/s | Blackwell |

### 3.12.3 NVLink拓扑设计

**DGX A100拓扑**：

```
8 × A100
│
├── NVSwitch 1 ←──→ GPU 0, 1, 2, 3, 4, 5, 6, 7
├── NVSwitch 2 ←──→ GPU 0, 1, 2, 3, 4, 5, 6, 7
└── NVSwitch 3 ←──→ GPU 0, 1, 2, 3, 4, 5, 6, 7

每个GPU 12链路
│ NVSwitch 1: 4链路
│ NVSwitch 2: 4链路
│ NVSwitch 3: 4链路
```

### 3.12.4 NVLink vs PCIe

| 特性 | NVLink | PCIe 4.0 x16 |
|------|--------|--------------|
| 带宽 | 600 GB/s (A100) | 32 GB/s |
| 延迟 | 极低 | 较高 |
| 扩展性 | 多GPU | 有限 |
| 适用场景 | GPU-GPU | GPU-CPU |

## 本章小结

本章系统介绍了NVIDIA GPU架构的发展历程和关键技术：

1. **九代架构演进**：从Fermi到Blackwell，每代架构都有重大技术创新

2. **Tensor Core演进**：从Volta首次引入到Blackwell第五代，性能不断提升

3. **CUDA编程模型**：Grid/Block/Thread层次，内存层次结构

4. **NVLink与NVSwitch**：从点对点互联到全互联集群

NVIDIA的技术领先地位来自：
- 持续架构创新
- 完整软件生态
- 统一编程模型
- 规模化部署验证

## 思考与练习

1. **架构演进**：比较Fermi和Blackwell架构的核心差异，分析这9代架构演进的驱动力是什么？

2. **Tensor Core分析**：分析Tensor Core从Volta到Blackwell的演进历程，每代新增了哪些特性和精度支持？

3. **CUDA理解**：解释CUDA编程中Grid、Block、Thread的层次关系，以及如何确定线程索引与数据的映射？

4. **性能优化思考**：如果一个CUDA程序性能不佳，请列举至少3个可能的优化方向（从内存访问、线程调度、计算模式等角度）。

5. **NVLink分析**：分析NVLink与PCIe的区别，为什么在多GPU系统中NVLink比PCIe更重要？

6. **架构选型思考**：假设你要构建一个大规模AI训练集群，请分析如何选择GPU架构和互联方案，考虑哪些因素？
