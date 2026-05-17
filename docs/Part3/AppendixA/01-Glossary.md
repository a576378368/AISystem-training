# 附录A：术语表

本附录整理了AI系统学习和工作中常用的专业术语中英文对照，按字母顺序排列，便于读者查阅和记忆。术语表涵盖了人工智能、机器学习、深度学习、硬件架构、编译器、推理系统等AI系统核心领域的专业词汇。

---

## A. 基础概念与通用术语

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| Artificial Intelligence | 人工智能 | AI | 研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的新兴技术科学 |
| Machine Learning | 机器学习 | ML | 人工智能的一个分支，研究如何让计算机从数据中自动学习并改进算法性能 |
| Deep Learning | 深度学习 | DL | 使用多层神经网络进行学习的机器学习方法，能够自动学习特征表示 |
| Neural Network | 神经网络 | NN | 模拟生物神经网络的计算模型，由大量神经元连接而成 |
| Artificial Neural Network | 人工神经网络 | ANN | 由人工神经元组成的计算系统，模仿生物神经网络的结构和功能 |
| Perceptron | 感知机 | | 最简单的人工神经元模型，是神经网络的基础组成单元 |
| Multi-Layer Perceptron | 多层感知器 | MLP | 由输入层、隐藏层和输出层组成的前馈神经网络 |
| Deep Neural Network | 深度神经网络 | DNN | 包含多个隐藏层的神经网络，具有强大的表征学习能力 |
| Convolutional Neural Network | 卷积神经网络 | CNN | 专门用于处理图像数据的神经网络，通过卷积操作提取空间特征 |
| Recurrent Neural Network | 循环神经网络 | RNN | 适合处理序列数据的神经网络，能够记住之前的信息 |
| Long Short-Term Memory | 长短期记忆网络 | LSTM | 一种特殊的RNN，通过门控机制解决长期依赖问题 |
| Transformer | 变换器 | | 基于注意力机制的神经网络架构，是大语言模型的基础 |
| Generative Pre-trained Transformer | 生成式预训练变换器 | GPT | 基于Transformer的大规模预训练语言模型 |
| Bidirectional Encoder Representations from Transformers | 变换器的双向编码器表示 | BERT | 基于Transformer的双向预训练语言模型 |

---

## B. 数学与优化基础

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| Gradient | 梯度 | | 多元函数在某点的偏导数向量，指向函数值增长最快的方向 |
| Gradient Descent | 梯度下降 | GD | 常用的优化算法，沿着梯度负方向更新参数以最小化损失函数 |
| Stochastic Gradient Descent | 随机梯度下降 | SGD | 每次使用单个样本或小批量样本计算梯度进行参数更新 |
| Learning Rate | 学习率 | | 梯度下降中参数更新的步长，控制模型学习的速度 |
| Loss Function | 损失函数 | | 衡量模型预测值与真实值之间差距的函数 |
| Mean Squared Error | 均方误差 | MSE | 回归任务常用的损失函数，计算预测值与真实值差的平方均值 |
| Cross Entropy | 交叉熵 | | 分类任务常用的损失函数，衡量两个概率分布的差异 |
| Activation Function | 激活函数 | | 向神经网络中引入非线性因素的函数，如ReLU、Sigmoid等 |
| ReLU | 线性整流函数 | | 常用的激活函数，f(x) = max(0, x) |
| Sigmoid | S型曲线函数 | | 将值映射到(0,1)区间的激活函数 |
| Softmax | Softmax函数 | | 将向量映射为概率分布的函数，常用于多分类输出层 |
| Back Propagation | 反向传播 | BP | 计算神经网络梯度的核心算法，通过链式法则反向传递误差 |
| Forward Propagation | 前向传播 | | 输入数据在神经网络中从输入层到输出层的计算过程 |
| Parameter | 参数 | | 神经网络中需要学习的权重和偏置 |
| Hyperparameter | 超参数 | | 训练前需要设置的参数，如学习率、批量大小等 |
| Over Fitting | 过拟合 | | 模型在训练数据上表现良好但在测试数据上泛化能力差 |
| Under Fitting | 欠拟合 | | 模型在训练数据和测试数据上都表现不佳 |
| Regularization | 正则化 | | 防止过拟合的技术，如L1、L2正则化、Dropout等 |
| Dropout | 丢弃法 | | 训练时随机丢弃部分神经元以防止过拟合 |
| Batch Normalization | 批归一化 | BatchNorm | 对批量数据归一化以加速训练稳定性的技术 |
| Layer Normalization | 层归一化 | LayerNorm | 对单个样本的所有特征进行归一化 |

---

## C. 硬件与芯片架构

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| Central Processing Unit | 中央处理器 | CPU | 计算机的核心处理单元，适合串行计算和复杂逻辑控制 |
| Graphics Processing Unit | 图形处理器 | GPU | 专为并行计算设计的处理器，擅长处理大规模并行任务 |
| Tensor Processing Unit | 张量处理器 | TPU | 谷歌专为深度学习设计的专用AI芯片 |
| Neural Processing Unit | 神经网络处理器 | NPU | 专为神经网络计算的AI芯片，如华为昇腾系列 |
| Field-Programmable Gate Array | 现场可编程门阵列 | FPGA | 可编程的硬件芯片，可根据需求配置为特定功能 |
| Application-Specific Integrated Circuit | 专用集成电路 | ASIC | 为特定应用定制的芯片，如Google TPU |
| Accelerator | 加速器 | | 专门用于加速特定计算的硬件设备 |
| SIMD | 单指令多数据流 | SIMD | 一种并行计算模式，一条指令处理多个数据 |
| SIMT | 单指令多线程 | SIMT | GPU采用的并行执行模式 |
| CUDA | 统一计算设备架构 | CUDA | NVIDIA推出的并行计算平台和编程模型 |
| Tensor Core | 张量核 | | GPU中专门用于加速矩阵运算的硬件单元 |
| NVLink | NVIDIA高速互联技术 | | GPU之间的高速数据传输通道 |
| Memory Hierarchy | 内存层次结构 | | 寄存器→L1缓存→L2缓存→L3缓存→主存→存储的层次结构 |
| Cache | 缓存 | | 位于CPU和主存之间的高速存储器 |
| Bandwidth | 带宽 | | 数据传输的速率，通常以GB/s为单位 |
| Throughput | 吞吐量 | | 单位时间内处理的数据量 |
| Latency | 延迟 | | 完成一次操作所需的时间 |
| Power Consumption | 功耗 | | 芯片工作时消耗的电功率 |
| Thermal Design Power | 热设计功率 | TDP | 芯片散热系统需要处理的的最大热量 |
| Die | 芯片核心 | | 芯片上集成电路的物理区域 |
| Compute Density | 计算密度 | | 单位面积或体积内的计算能力 |

---

## D. 编译器与中间表示

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| Compiler | 编译器 | | 将高级语言转换为机器语言的程序 |
| Source-to-Source Compiler | 源到源编译器 | | 将一种高级语言转换为另一种高级语言的编译器 |
| Intermediate Representation | 中间表示 | IR | 编译器中用于表示源代码的抽象数据结构 |
| Abstract Syntax Tree | 抽象语法树 | AST | 表示程序语法结构的树形数据结构 |
| Three-Address Code | 三地址码 | TAC/3AC | 形如x = y op z的简单赋值语句序列 |
| Static Single Assignment | 静态单赋值 | SSA | 每个变量只被赋值一次的IR形式，便于优化 |
| Lexical Analysis | 词法分析 | | 编译器前端阶段，将源代码分解为词素序列 |
| Syntax Analysis | 语法分析 | | 编译器前端阶段，根据语法规则分析词素序列 |
| Semantic Analysis | 语义分析 | | 编译器前端阶段，检查程序的语义正确性 |
| Code Generation | 代码生成 | | 编译器后端阶段，将IR转换为目标机器代码 |
| Optimization | 优化 | | 改进代码性能的过程，包括多种优化技术 |
| Loop Optimization | 循环优化 | | 针对循环结构的优化，如循环展开、循环分块等 |
| Constant Folding | 常量折叠 | | 编译时计算常量表达式的优化技术 |
| Dead Code Elimination | 死代码消除 | | 删除不会被执行的代码的优化技术 |
| Common Subexpression Elimination | 公共子表达式消除 | CSE | 消除重复计算的优化技术 |
| Inlining | 内联 | | 将函数调用替换为函数体的优化技术 |
| Register Allocation | 寄存器分配 | | 决定变量如何映射到硬件寄存器的过程 |
| Instruction Scheduling | 指令调度 | | 重新排列指令顺序以提高执行效率 |
| Data Flow Analysis | 数据流分析 | | 分析数据在程序中的传递和变化 |

---

## E. AI编译器与框架

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| AI Compiler | AI编译器 | | 专门用于深度学习模型的编译器 |
| Operator Fusion | 算子融合 | | 将多个连续算子合并为一个算子的优化技术 |
| Graph Optimization | 图优化 | | 对计算图进行结构优化和等价变换 |
| Kernel | 内核/核函数 | | 在硬件上执行的具体计算实现 |
| Auto Tuning | 自动调优 | | 自动搜索最优执行参数的技术 |
| AutoTVM | 自动TVM | | TVM的自动调优系统 |
| Ansor | | | TVM的第二代自动调优系统 |
| Meta Schedule | 元调度 | | TVM的新一代自动调优框架 |
| XLA | 加速线性代数 | XLA | TensorFlow的编译器 |
| Glow | | | Facebook的AI编译器 |
| MLIR | 多级中间表示 | MLIR | LLVM旗下的通用编译器框架 |
| TVM | 端到端优化编译器 | TVM | Apache TVM深度学习编译器 |
| Relay IR | Relay中间表示 | | TVM的神经网络中间表示 |
| TorchScript | | | PyTorch的模型导出格式 |
| ONNX | 开放神经网络交换 | ONNX | 跨框架模型交换格式 |
| SafeTensors | 安全张量格式 | | 高效安全的模型序列化格式 |
| Framework | 框架 | | 提供AI开发基础结构和工具的软件系统 |
| Programming Paradigm | 编程范式 | | 编程的基本方式和风格 |
| Dynamic Graph | 动态计算图 | | 运行时动态构建的计算图，灵活但效率较低 |
| Static Graph | 静态计算图 | | 编译时预先构建的计算图，效率高但灵活性低 |
| JIT Compilation | 即时编译 | JIT | 运行时动态编译的技术 |

---

## F. 自动微分与计算图

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| Automatic Differentiation | 自动微分 | AD | 自动计算函数导数的技术 |
| Computational Graph | 计算图 | | 表示计算依赖关系的有向无环图 |
| Data Flow Graph | 数据流图 | DFG | 表示数据在计算节点间流动的图 |
| Control Flow | 控制流 | | 程序执行顺序的控制结构 |
| Differential | 微分 | | 描述函数局部线性变化的数学概念 |
| Derivative | 导数 | | 描述函数变化率的概念 |
| Partial Derivative | 偏导数 | | 多元函数对单个变量的导数 |
| Chain Rule | 链式法则 | | 计算复合函数导数的法则 |
| Jacobian | 雅可比矩阵 | | 多元函数一阶偏导数构成的矩阵 |
| Gradient Accumulation | 梯度累积 | | 多个小批量梯度累加模拟大批量训练 |
| Mixed Precision | 混合精度 | | 同时使用多种精度进行训练和推理 |
| FP16 | 半精度浮点 | | 16位浮点数格式 |
| FP32 | 单精度浮点 | | 32位浮点数格式 |
| FP64 | 双精度浮点 | | 64位浮点数格式 |
| BF16 | BFloat16 | | 谷歌设计的16位浮点格式，指数位与FP32相同 |
| Symbolic Execution | 符号执行 | | 用符号值而非具体值执行程序的技术 |

---

## G. 模型压缩与量化

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| Model Compression | 模型压缩 | | 减小模型尺寸和计算量的技术总称 |
| Quantization | 量化 | | 将高精度数据转换为低精度表示的技术 |
| Weight Quantization | 权重量化 | | 仅对模型权重进行量化 |
| Dynamic Quantization | 动态量化 | | 推理时动态确定量化参数 |
| Static Quantization | 静态量化 | | 预先确定量化参数的量化方法 |
| Quantization-Aware Training | 量化感知训练 | QAT | 在训练中模拟量化效应 |
| Post-Training Quantization | 训练后量化 | PTQ | 模型训练完成后进行量化 |
| INT8 | 8位整数 | | 常用的低精度表示格式 |
| Pruning | 剪枝 | | 删除网络中不重要的连接或神经元 |
| Structured Pruning | 结构化剪枝 | | 按结构删除神经元组 |
| Unstructured Pruning | 非结构化剪枝 | | 随机删除单个连接 |
| Knowledge Distillation | 知识蒸馏 | | 将大模型知识迁移到小模型的技术 |
| Teacher-Student | 教师-学生 | | 知识蒸馏中的大模型和小模型 |
| Model Sparse | 模型稀疏 | | 权重中包含大量零值的模型状态 |
| Low-Rank Factorization | 低秩分解 | | 用低秩矩阵近似原始权重矩阵 |
| Neural Architecture Search | 神经网络结构搜索 | NAS | 自动搜索最优网络结构的技术 |

---

## H. 推理系统与部署

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| Inference | 推理 | | 使用训练好的模型进行预测的过程 |
| Inference System | 推理系统 | | 支持模型部署和推理请求处理的完整系统 |
| Inference Engine | 推理引擎 | | 执行模型推理的核心组件 |
| Model Serving | 模型服务 | | 将模型部署为服务供外部调用 |
| Model Deployment | 模型部署 | | 将训练好的模型发布到生产环境 |
| Online Service | 在线服务 | | 实时处理请求的服务模式 |
| Batch Inference | 批量推理 | | 一次处理多个输入的推理方式 |
| Real-Time Inference | 实时推理 | | 低延迟要求的即时推理 |
| Request | 请求 | | 发送给推理系统的输入数据 |
| Response | 响应 | | 推理系统返回的预测结果 |
| Preprocessing | 预处理 | | 推理前对输入数据进行的处理 |
| Postprocessing | 后处理 | | 推理后对输出结果进行的处理 |
| Batching | 批处理 | | 将多个请求合并处理的机制 |
| Dynamic Batching | 动态批处理 | | 根据条件动态调整批大小的技术 |
| Scheduling | 调度 | | 决定任务执行顺序和时间的技术 |
| Load Balancing | 负载均衡 | | 分配推理请求到多个推理实例 |
| Model Versioning | 模型版本管理 | | 管理不同版本模型的系统 |
| A/B Testing | A/B测试 | | 比较两个模型版本性能的技术 |
| Canary Deployment | 金丝雀部署 | | 渐进式发布新模型的策略 |
| Shadow Mode | 影子模式 | | 新模型并行运行但不返回结果 |
| Model Registry | 模型注册表 | | 集中存储和管理模型的地方 |

---

## I. 分布式训练与并行计算

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| Distributed Training | 分布式训练 | | 在多个计算设备上进行模型训练 |
| Data Parallelism | 数据并行 | | 不同设备处理不同数据，持有相同模型副本 |
| Model Parallelism | 模型并行 | | 将模型拆分到不同设备上 |
| Pipeline Parallelism | 流水并行 | | 不同设备处理模型的不同阶段形成流水线 |
| Tensor Parallelism | 张量并行 | | 将张量拆分到多个设备上计算 |
| Hybrid Parallelism | 混合并行 | | 结合多种并行策略 |
| Synchronous Training | 同步训练 | | 所有设备同步更新参数 |
| Asynchronous Training | 异步训练 | | 设备异步更新参数，可能存在陈旧梯度问题 |
| Parameter Server | 参数服务器 | | 集中管理模型参数的服务器 |
| AllReduce | 全规约 | | 分布式计算中汇总所有节点数据 |
| Collective Communication | 集体通信 | | 多个节点间的通信操作 |
| Point-to-Point Communication | 点对点通信 | | 两个节点间的直接通信 |
| Gradient Synchronization | 梯度同步 | | 同步各设备计算的梯度 |
| Local Batch Size | 本地批量大小 | | 单个设备上的批量大小 |
| Global Batch Size | 全局批量大小 | | 所有设备批量大小的总和 |
| Worker | 工作节点 | | 执行计算任务的节点 |
| Master | 主节点 | | 协调其他节点工作的节点 |
| Elastic Training | 弹性训练 | | 支持动态添加删除节点 |
| Fault Tolerance | 容错 | | 节点故障时继续训练的能力 |
| Checkpoint | 检查点 | | 保存训练状态用于恢复 |

---

## J. 性能指标与评测

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| Accuracy | 准确率 | | 正确预测样本占总样本的比例 |
| Precision | 精确率 | | 预测为正的样本中真正为正的比例 |
| Recall | 召回率 | | 真正为正的样本中被正确预测的比例 |
| F1 Score | F1分数 | | 精确率和召回率的调和平均 |
| Confusion Matrix | 混淆矩阵 | | 展示分类预测与真实标签关系的矩阵 |
| Throughput | 吞吐量 | | 单位时间内处理的样本数量 |
| Latency | 延迟 | | 一次推理所需的时间 |
| Tail Latency | 尾部延迟 | | 高百分位的延迟，如P99延迟 |
| Frames Per Second | 每秒帧数 | FPS | 图像/视频处理中的性能指标 |
| Queries Per Second | 每秒查询数 | QPS | 服务处理请求的速率 |
| Mean Time to Recovery | 平均恢复时间 | MTTR | 系统故障后恢复的平均时间 |
| Model Size | 模型大小 | | 模型占用的存储空间 |
| Memory Footprint | 内存占用 | | 模型运行时占用的内存量 |
| GPU Utilization | GPU利用率 | | GPU计算资源的使用程度 |
| Roofline Model | 屋顶线模型 | | 评估计算性能的理论模型 |
| Benchmark | 基准测试 | | 用于性能评估的标准测试 |
| Speedup | 加速比 | | 优化后相比优化前的性能提升比例 |
| Amdahl's Law | 阿姆达尔定律 | | 并行计算加速比的理论上限 |

---

## K. 系统架构与运维

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| Container | 容器 | | 轻量级的虚拟化技术，封装应用及其依赖 |
| Docker | | | 流行的容器化平台 |
| Kubernetes | | | 容器编排平台，用于自动化部署和管理容器化应用 |
| Microservices | 微服务 | | 将应用拆分为小型独立服务的架构风格 |
| Service Mesh | 服务网格 | | 微服务间通信的基础设施层 |
| API Gateway | API网关 | | 作为系统统一入口的网关服务 |
| Load Balancer | 负载均衡器 | | 分发请求到多个后端服务的组件 |
| Reverse Proxy | 反向代理 | | 代理服务器接受客户端请求并转发给内部服务器 |
| Service-Level Agreement | 服务等级协议 | SLA | 服务提供商与客户之间的服务质量承诺 |
| Service-Level Objective | 服务等级目标 | SLO | 服务的具体性能目标 |
| Continuous Integration | 持续集成 | CI | 频繁合并代码并自动测试的实践 |
| Continuous Delivery | 持续交付 | CD | 代码经过测试后可以随时部署的生产准备状态 |
| Monitoring | 监控 | | 追踪系统状态和性能的活动 |
| Observability | 可观测性 | | 通过数据理解系统内部状态的能力 |
| Telemetry | 遥测 | | 远程收集系统数据的技术 |
| Logging | 日志 | | 记录系统事件的信息 |
| Tracing | 追踪 | | 跟踪请求在系统中的完整路径 |
| Incident | 事件 | | 导致服务中断或性能下降的问题 |
| On-Call | 值班 | | 随时响应系统告警的责任 |
| Runbook | 运行手册 | | 处理常见问题的标准操作流程 |
| Single Point of Failure | 单点故障 | | 系统中的单一故障点 |
| High Availability | 高可用 | HA | 系统持续运行的可用性能力 |

---

## L. 大模型相关术语

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| Large Language Model | 大语言模型 | LLM | 参数规模巨大的语言模型 |
| Foundation Model | 基础模型 | | 在大规模数据上预训练的大型模型 |
| Pre-training | 预训练 | | 在大规模无标注数据上进行的自监督学习 |
| Fine-tuning | 微调 | | 在特定任务上调整预训练模型 |
| Instruction Tuning | 指令微调 | | 使用指令-响应对微调模型 |
| Reinforcement Learning from Human Feedback | 人类反馈强化学习 | RLHF | 利用人类反馈优化模型 |
| Prompt Engineering | 提示工程 | | 设计输入提示以获得更好输出 |
| Prompt | 提示 | | 输入给模型的文本引导 |
| Few-Shot Learning | 小样本学习 | | 从少量样本中学习的能力 |
| Zero-Shot Learning | 零样本学习 | | 无需训练样本即可完成任务 |
| In-Context Learning | 上下文学习 | | 在提示中提供示例的学习方式 |
| Chain-of-Thought | 思维链 | | 展示推理过程的提示技术 |
| Temperature | 温度 | | 控制生成随机性的采样参数 |
| Top-k Sampling | Top-k采样 | | 只考虑前k个最可能token的采样方法 |
| Nucleus Sampling | 核心采样 | | 只考虑累计概率达到阈值的token |
| Context Length | 上下文长度 | | 模型一次能处理的最大token数 |
| Attention Mechanism | 注意力机制 | | 让模型关注输入不同部分的技术 |
| Self-Attention | 自注意力 | | 输入序列内部元素间的注意力计算 |
| Multi-Head Attention | 多头注意力 | | 并行计算多组注意力 |
| Positional Encoding | 位置编码 | | 为序列中的位置信息进行编码 |
| Token | 词元 | | 文本被分割的最小单位 |
| Tokenizer | 分词器 | | 将文本分割为token的工具 |
| Embedding | 嵌入 | | 将离散token映射为连续向量 |
| Vocabulary | 词表 | | 模型认识的全部token集合 |
| Vocabulary Size | 词表大小 | | 词表中token的数量 |
| Generation | 生成 | | 模型根据输入产生输出的过程 |
| Inference | 推理 | | 模型产生输出的过程（也指代模型本身） |
| Hallucination | 幻觉 | | 模型产生看似合理但错误的内容 |
| Alignment | 对齐 | | 使模型行为符合人类意图和价值观 |
| Scaling Law | 缩放定律 | | 模型性能随规模变化的规律 |

---

## M. 芯片架构与设计

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| Instruction Set Architecture | 指令集架构 | ISA | 处理器支持的指令集合和功能定义 |
| Reduced Instruction Set Computer | 精简指令集计算机 | RISC | 简化的指令集设计理念 |
| Complex Instruction Set Computer | 复杂指令集计算机 | CISC | 复杂指令集设计理念 |
| Pipelining | 流水线 | | 将指令执行分为多个阶段并行处理 |
| Out-of-Order Execution | 乱序执行 | | 不按程序顺序执行指令以提高效率 |
| Branch Prediction | 分支预测 | | 预测程序分支走向的技术 |
| Superscalar | 超标量 | | 同时发射多条指令的处理器 |
| Very Long Instruction Word | 超长指令字 | VLIW | 一条指令包含多个操作的架构 |
| Systolic Array | 脉动阵列 | | 專用於矩陣運算的硬體陣列結構 |
| Die Area | 芯片面积 | | 芯片核心区域的大小 |
| Process Node | 制程节点 | | 芯片制造工艺的尺寸标准 |
| Clock Frequency | 时钟频率 | | 处理器运行的速度 |
| IPC | 每指令周期数 | IPC | 每周期执行的指令数 |
| Utilization | 利用率 | | 硬件资源实际使用程度 |
| Roofline | 屋顶线 | | 展示计算性能上限的模型 |
| Arithmetic Intensity | 算术强度 | | 每字节内存传输的计算量 |
| Off-Chip Memory | 片外内存 | | 芯片外部的内存 |
| On-Chip Memory | 片上内存 | | 芯片内部集成的内存 |
| scratchpad | 便笺式内存 | | 片上高速内存 |
| Data Reuse | 数据复用 | | 多次使用同一数据减少内存访问 |

---

## N. 其他重要术语

| 英文术语 | 中文术语 | 缩写 | 释义 |
|---------|---------|------|------|
| API | 应用程序接口 | API | 定义软件组件交互方式的接口 |
| SDK | 软件开发包 | SDK | 帮助开发特定应用的工具集合 |
| End-to-End | 端到端 | | 从输入到输出的完整过程 |
| Edge Computing | 边缘计算 | | 在数据源附近进行计算的技术 |
| Cloud Computing | 云计算 | | 通过网络提供计算资源的服务 |
| Data Center | 数据中心 | | 放置计算设备和存储设备的设施 |
| RDMA | 远程直接内存访问 | RDMA | 直接访问远程内存的技术 |
| InfiniBand | | |高速网络互连技术 |
| Ethernet | 以太网 | | 常用的局域网技术 |
| PCIe | 高速外部设备互连总线 | PCIe | 连接硬件设备的接口标准 |
| Virtual Machine | 虚拟机 | | 虚拟化的计算环境 |
| Kernel-Based Virtual Machine | 基于内核的虚拟机 | KVM | Linux内核的虚拟化技术 |
| SR-IOV | 单根I/O虚拟化 | | 允许虚拟机直接访问硬件的技术 |
| Passthrough | 直通 | | 虚拟机直接使用物理设备的技术 |
| Topology | 拓扑 | | 系统或网络的结构布局 |
| Fat Tree | 胖树 | | 数据中心常用网络拓扑结构 |
| Dragonfly | 蜻蜓 | | 一种高扩展性网络拓扑 |
| Modularity | 模块化 | | 将系统分解为独立模块的设计方法 |
| Abstraction | 抽象 | | 隐藏细节只暴露关键特征的概念 |
| Interface | 接口 | | 定义组件交互方式的边界 |
| Implementation | 实现 | | 具体代码和逻辑的完成 |
| Specification | 规格说明 | | 对系统功能的详细描述 |
| Verification | 验证 | | 确认系统实现符合规格的过程 |
| Validation | 确认 | | 确保系统满足用户需求的过程 |
| Benchmark | 基准测试 | | 用于比较性能的标准测试 |
| Profiling | 性能分析 | | 分析程序性能特征的活动 |
| Debugging | 调试 | | 定位和修复程序错误的过程 |
| Hotspot | 热点 | | 程序中消耗最多时间的部分 |
| Bottleneck | 瓶颈 | | 限制整体性能的关键点 |

---

## 附录使用说明

本术语表按照主题分类组织，便于读者在学习和工作中快速查阅。建议读者：

1. **通读理解**：首先通读一遍，建立对AI系统各领域术语的初步认识
2. **重点记忆**：根据学习进度，重点记忆当前章节相关的术语
3. **查阅参考**：在实际学习和工作中遇到陌生术语时，随时查阅
4. **实践应用**：在写作和交流中主动使用这些术语，加深记忆

术语表会持续更新补充，欢迎读者提供宝贵意见和改进建议。