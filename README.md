# AI系统工程师培训教程

[![GitHub Stars](https://img.shields.io/github/stars/a576378368/AISystem-training?style=flat-square)](https://github.com/a576378368/AISystem-training)
[![License](https://img.shields.io/badge/License-Apache--2.0%20%26%20MIT-blue.svg)](https://github.com/a576378368/AISystem-training/blob/main/LICENSE)

**仓库地址**：[https://github.com/a576378368/AISystem-training](https://github.com/a576378368/AISystem-training)

## 简介

AI系统工程师培训教程，专为零基础新人设计，旨在帮助半导体公司算法设计部门的工程师系统掌握AI系统的核心知识。教程内容基于开源项目AISystem整理优化，按照学习路径由浅入深编排，图文并茂，概念清晰。

## 学习路径

| 阶段 | 章节 | 内容 |
|------|------|------|
| 第一阶段 | Chap01-03 | 预备知识（计算机基础/Python编程/数学基础） |
| 第二阶段 | Chap04-05 | AI基础（概述/神经网络） |
| 第二阶段 | Chap06 | AI芯片体系架构 |
| 第二阶段 | Chap07 | AI编译器原理 |
| 第二阶段 | Chap08 | 推理系统与引擎 |
| 第二阶段 | Chap09-10 | AI框架核心/大模型与集群 |
| 第三阶段 | Appendix | 附录（术语表/参考文献/实践项目） |

## 内容结构

```
AISystem-training/
├── Part1/              # 第一阶段：预备知识
│   ├── Chap01/         # 计算机基础
│   ├── Chap02/         # Python编程
│   └── Chap03/         # 数学基础
├── Part2/              # 第二阶段：AI系统核心
│   ├── Chap04/         # AI系统概述
│   ├── Chap05/         # 神经网络基础
│   ├── Chap06/         # AI芯片体系架构
│   ├── Chap07/         # AI编译器原理
│   ├── Chap08/         # 推理系统与引擎
│   ├── Chap09/         # AI框架核心
│   └── Chap10/         # 大模型与集群
├── Part3/              # 第三阶段：附录
├── PPT/                # 教学PPT
├── PPT-Markdown/       # PPT内容Markdown版本
└── images/             # 图片资源
```

## AI系统全栈知识图谱

```
应用层     → AI系统全栈、算法框架、系统架构
框架层     → 神经网络、自动微分、计算图、分布式训练
编译器层   → 传统编译器、AI编译器、前后端优化、算子融合
硬件层     → GPU原理、NPU/TPU架构、编程模式
推理系统   → 模型压缩、推理优化、Kernel优化
大模型     → LLM架构、AI集群、并行策略
```

## 快速开始

```bash
# 克隆仓库
git clone git@github.com:a576378368/AISystem-training.git
cd AISystem-training

# 安装依赖
pip install -r requirements.txt

# 构建HTML文档
sphinx-build -b html . _build/html

# 构建PDF（需要LaTeX）
sphinx-build -b pdf . _build/pdf
```

## 在线文档

访问 GitHub Pages 查看完整文档：

👉 **https://a576378368.github.io/AISystem-training**

## 主要内容

### 第一阶段：预备知识

- **Chap01 计算机基础**：进制与编码、CPU与内存、程序运行原理
- **Chap02 Python编程**：Python基础语法、NumPy基础、PyTorch基础
- **Chap03 数学基础**：线性代数、概率统计、优化基础

### 第二阶段：AI系统核心

- **Chap04 AI系统概述**：AI系统全栈、算法框架、系统架构
- **Chap05 神经网络基础**：神经元模型、CNN/RNN/Transformer、量化
- **Chap06 AI芯片体系架构**：AI芯片分类、GPU原理、主流芯片、编程模式
- **Chap07 AI编译器原理**：传统编译器、AI编译器、前后端优化
- **Chap08 推理系统与引擎**：推理流程、模型压缩、推理优化
- **Chap09 AI框架核心**：自动微分、计算图、分布式训练
- **Chap10 大模型与集群**：LLM架构、AI集群、并行策略

### 第三阶段：附录

- **术语表**：AI系统常用术语中英文对照
- **参考文献**：书籍、在线课程、论文、文档推荐
- **实践项目**：神经网络实现、模型量化部署、分布式训练、AI编译器探索

## 原始项目

本教程基于 [AISystem](https://github.com/Infrasys-AI/AISystem)（Apache-2.0 & MIT License）整理优化。

感谢 [zomi](https://github.com/chenzomi12) 等作者的贡献！

## 许可

本教程采用 Apache-2.0 & MIT 双许可证开源。

- [LICENSE](LICENSE) - 主许可证（内容）
- [LICENSE-CODE](LICENSE-CODE) - 代码许可证

## 贡献

欢迎提交 Issue 和 Pull Request！