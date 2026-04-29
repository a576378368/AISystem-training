# Chap06 AI芯片体系架构

## 本章学习目标

1. 理解AI芯片的基本概念和分类
2. 掌握GPU的工作原理和架构
3. 了解主流AI芯片（TPU/昇腾/寒武纪等）
4. 理解DSA/SIMT/SIMD等编程模式

## 前置知识

- Chap04 AI系统概述
- Chap05 神经网络基础

## 内容导航

```{toctree}
:maxdepth: 2

01-Chip-Base
02-GPU-Base
03-NVIDIA
04-Abroad-Chips
05-Domestic-Chips
06-Programming-Model
```

## 本章小结

本章系统介绍了AI芯片体系架构的核心内容，主要包括以下方面：

**AI芯片基础**：AI芯片是专门为加速人工智能应用中的矩阵计算、卷积运算等核心任务而设计的处理器，采用领域专用架构（DSA）优化。AI芯片可分为CPU、GPU、FPGA、ASIC四大类，按应用场景可分为云端芯片和边缘端芯片。

**GPU原理**：GPU从图形专用演进为通用并行计算平台，其核心是SIMT（单指令多线程）执行模型。GPU通过大规模线程超配掩盖内存延迟，实现高吞吐量计算。Tensor Core专门加速深度学习矩阵运算，支持混合精度训练。

**NVIDIA GPU架构**：NVIDIA GPU从Fermi到Blackwell经历九代演进，核心技术包括CUDA编程模型、Tensor Core、NVLink高速互联。CUDA通过Grid/Block/Thread层次组织线程，NVLink实现多GPU高效互联。

**国外AI芯片**：Google TPU采用脉动阵列架构专门优化矩阵运算，经历v1到v4四代演进。Tesla DOJO专为自动驾驶训练设计。Intel Gaudi面向云端AI训练市场。

**国内AI芯片**：华为昇腾采用达芬奇架构，昇腾910面向云端训练，昇腾310面向边缘推理，CANN软件栈提供完整工具链。寒武纪MLU采用MLU Core+Memory Core设计，BANG C编程语言。燧原、壁仞等企业也在各自领域有技术积累。

**芯片编程模式**：SIMD通过单指令多数据实现并行，SIMT在硬件层面支持线程独立执行。异构计算通过CPU+GPU/FPGA/ASIC协同工作，超异构是发展趋势。

## 思考与练习

1. **综合分析**：对比分析CPU、GPU、FPGA、ASIC四大芯片架构在AI计算场景下的优劣势，并说明各自的典型应用场景。

2. **GPU理解**：解释GPU的SIMT执行模型与CPU的SIMD执行模型的核心区别。为什么GPU比CPU更适合执行深度学习工作负载？

3. **架构演进**：分析NVIDIA GPU从Fermi到Blackwell九代架构演进的主要驱动力。每代架构的核心技术创新是什么？

4. **TPU分析**：Google TPU采用的脉动阵列架构与NVIDIA GPU采用的架构有什么本质区别？脉动阵列的优势和局限性是什么？

5. **国产芯片**：分析华为昇腾和寒武纪这两家国内AI芯片厂商的技术路线差异。国产AI芯片在软件生态建设方面面临哪些挑战？

6. **编程模式**：比较SIMD、SIMT和DSA三种编程模式的特点和适用场景。如果你要开发一个深度学习应用，会如何选择合适的硬件和编程模式？

7. **异构计算**：解释什么是超异构计算。为什么说超异构是未来计算架构的发展方向？实现超异构计算需要解决哪些技术挑战？
