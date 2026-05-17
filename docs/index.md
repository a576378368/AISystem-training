---
html_theme_options:
  full_width: true
---

# AI系统工程师培训教程

## 关于本教程

### 教材简介

本教程是专为**零基础新人**设计的AI系统培训教材，旨在帮助半导体公司算法设计部门的工程师系统掌握AI系统的核心知识。教程内容基于开源项目AISystem整理优化，按照**学习路径**由浅入深编排，图文并茂，概念清晰。

### 目标读者

- 零基础新人（无AI背景）
- 从半导体硬件转向AI软件的工程师
- 希望系统学习AI全栈知识的人员

### 学习路径

**前置知识要求**：
- 基本的编程概念
- 高中数学基础
- 对AI有初步兴趣

### 内容结构

| 阶段 | 章节 | 内容 | 目标字数 |
|------|------|------|---------|
| 第一阶段 | Chap01-03 | 预备知识（计算机/Python/数学） | 22万 |
| 第二阶段 | Chap04-05 | AI基础（概述/神经网络） | 16万 |
| 第二阶段 | Chap06 | AI芯片体系架构 | 20万 |
| 第二阶段 | Chap07 | AI编译器原理 | 20万 |
| 第二_phase | Chap08 | 推理系统与引擎 | 25万 |
| 第二阶段 | Chap09-10 | AI框架与大模型 | 30万 |
| 第三阶段 | Appendix | 附录（术语表/实践项目） | 5万+ |
| **总计** | | | **208万+** |

---

```{toctree}
:maxdepth: 2
:caption: 第一阶段：AI基础入门

Part1/Chap01/README
Part1/Chap02/README
Part1/Chap03/README
```

```{toctree}
:maxdepth: 2
:caption: 第二阶段：AI系统核心

Part2/Chap04/README
Part2/Chap05/README
Part2/Chap06/README
Part2/Chap07/README
Part2/Chap08/README
Part2/Chap09/README
Part2/Chap10/README
```

```{toctree}
:maxdepth: 2
:caption: 第三阶段：附录

Part3/AppendixA/README
Part3/AppendixB/README
Part3/AppendixC/README
```

---

## 章节预览

### 第一阶段：预备知识

这一阶段为零基础读者提供必要的计算机基础、编程基础和数学基础。

- **Chap01 计算机基础**：进制与编码、CPU与内存、程序运行原理
- **Chap02 Python编程**：Python基础语法、NumPy基础、PyTorch基础
- **Chap03 数学基础**：线性代数、概率统计、优化基础

### 第二阶段：AI系统核心

这一阶段按照AI系统的层次结构，从上到下详细介绍各个组件。

- **Chap04 AI系统概述**：AI系统全栈、算法框架、系统架构
- **Chap05 神经网络基础**：神经元模型、网络结构、训练方法
- **Chap06 AI芯片体系架构**：AI芯片分类、GPU原理、主流芯片、编程模式
- **Chap07 AI编译器原理**：传统编译器、AI编译器、前后端优化
- **Chap08 推理系统与引擎**：推理流程、模型压缩、推理优化
- **Chap09 AI框架核心**：自动微分、计算图、分布式训练
- **Chap10 大模型与集群**：大模型架构、AI集群、分布式系统

### 第三阶段：附录

- **术语表**：AI系统常用术语中英文对照
- **参考文献**：学习资料推荐
- **实践项目**：动手练习题目

---

## 如何使用本教程

### 学习建议

1. **按顺序学习**：建议按照章节顺序由浅入深学习
2. **理论结合实践**：每章配有小结与思考题，帮助巩固知识
3. **图文结合**：教程配有大量图片辅助理解复杂概念

### 编译方式

本教程使用Sphinx编译，支持多种输出格式：

```bash
# 安装依赖
pip install -r requirements.txt

# 编译为HTML
sphinx-build -b html . _build/html

# 编译为PDF（需要LaTeX）
sphinx-build -b pdf . _build/pdf
```

---

## 版权声明

本教程仅供企业内部培训使用，内容基于开源项目AISystem整理。
