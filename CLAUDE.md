# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chinese AI System Engineer training tutorial built with Sphinx. It is a documentation-only repository (no Python code to build/test) organized into three learning phases:

- **Part1**: Prerequisites (Computer basics, Python, Math)
- **Part2**: AI System Core (AI overview, Neural networks, AI chips, AI compilers, Inference systems, AI frameworks, LLMs)
- **Part3**: Appendices (Glossary, References, Practice projects)

## Build Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build HTML documentation
sphinx-build -b html . _build/html

# Build PDF (requires LaTeX)
sphinx-build -b pdf . _build/pdf

# Clean build artifacts
sphinx-build -b html . _build/html -E
```

## Architecture

The tutorial uses Sphinx with:
- **myst-parser** for Markdown content
- **sphinx-book-theme** for the theme
- **myst-nb** for Jupyter notebook support
- Multi-language support (language: cn)

Content structure:
- `index.md` - Master document with toctree
- `conf.py` - Sphinx configuration
- `Part1/`, `Part2/`, `Part3/` - Chapter directories with numbered Markdown files
- `_static/` - Static assets (images, cover)
- `_templates/` - Sphinx templates

## Content Structure

### AI System Learning Path (Top-Down)

The tutorial follows a top-down approach covering the full AI system stack:

| Layer | Chapters | Topics |
|-------|----------|--------|
| 应用层 | Chap04 | AI系统全栈、算法框架、系统架构 |
| AI框架层 | Chap05, Chap09 | 神经网络基础、自动微分、计算图、分布式训练 |
| AI编译器层 | Chap07 | 传统编译器、AI编译器、前后端优化、算子融合 |
| 硬件层 | Chap06 | GPU原理、NPU/TPU架构、编程模式 |
| 推理系统 | Chap08 | 模型压缩、推理优化、Kernel优化 |
| 大模型 | Chap10 | LLM架构、AI集群、并行策略 |

### Part1 - Prerequisites
- **Chap01**: 进制编码、CPU与内存、程序运行原理
- **Chap02**: Python基础、NumPy、PyTorch基础
- **Chap03**: 线性代数、概率统计、优化基础

### Part2 - AI System Core
- **Chap04**: AI系统概述、全栈架构、设计目标
- **Chap05**: 神经元模型、CNN/RNN/Transformer、量化
- **Chap06**: AI芯片分类、GPU架构、主流芯片(NVIDIA/昇腾)、编程模式
- **Chap07**: 传统编译器、AI编译器(TVM/XLA/Glow/MLIR)、前端优化、后端优化
- **Chap08**: 推理系统、模型压缩(量化/剪枝/蒸馏)、模型转换(ONNX)、推理优化
- **Chap09**: 自动微分、计算图、分布式训练
- **Chap10**: LLM架构、AI集群、并行策略(数据并行/模型并行/混合并行)

### Part3 - Appendices
- **AppendixA**: 术语表 - AI系统常用术语中英文对照
- **AppendixB**: 参考文献 - 书籍、在线课程、论文、文档推荐
- **AppendixC**: 实践项目 - 神经网络实现、模型量化部署、分布式训练、AI编译器探索

## File Naming Conventions

Chapter content files use numbered prefixes (e.g., `01-Introduction.md`, `02-Architecture.md`). Some files may have macOS backup artifacts (`.!*.` files) that should be ignored.

## Key Technical Concepts

- **Multi-layer IR**: AI compilers use layered intermediate representations (e.g., TVM's relay IR, XLA's HLO)
- **Operator Fusion**: Critical optimization that merges consecutive operators to reduce memory access
- **AutoTuning**: Automatic kernel optimization (AutoTVM, Ansor, Meta Schedule)
- **Model Formats**: ONNX, TorchScript, SafeTensors for model interchange
- **Quantization**: FP32→INT8/FP16 for inference acceleration