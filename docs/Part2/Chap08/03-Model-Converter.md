# 03 模型转换与优化

## 学习目标

1. 理解模型转换的概念和必要性
2. 掌握主流模型格式（ONNX、TorchScript、SafeTensors等）的特点和使用场景
3. 了解计算图的基本概念和表示方法
4. 掌握图优化的主要技术（算子融合、布局转换、算子替换等）
5. 理解中间表示（IR）的设计和作用

## 3.1 模型转换概述

### 3.1.1 什么是模型转换

**模型转换（Model Conversion/Export）**是将训练好的模型从一种格式转换为另一种格式的过程。在深度学习应用中，模型训练和模型部署通常在不同的环境和平台上进行，模型转换就是连接训练与部署的桥梁。

**为什么需要模型转换？**

1. **训练框架多样性**：目前存在多种深度学习训练框架，如PyTorch、TensorFlow、MindSpore、PaddlePaddle等。每个框架有自己独特的模型格式和API。训练时通常选择研究友好的框架（如PyTorch），但部署时可能需要转换到其他框架。

2. **部署环境差异**：训练环境和推理环境往往不同。训练可能在高性能GPU集群上进行，使用Python接口；但部署可能在不同的硬件平台（如移动端CPU、嵌入式设备、专用AI芯片）上，需要不同的运行时环境。

3. **优化需求**：训练框架的模型表示通常包含训练相关的操作（如梯度计算、优化器状态等），这些在推理时不需要。转换过程可以清理这些冗余，生成更高效的推理模型。

4. **跨平台支持**：同一个模型可能需要部署到多种平台。通过统一的中间格式，可以一次转换，多次部署。

### 3.1.2 模型转换的挑战

模型转换面临多方面的技术挑战：

**算子语义差异**

不同框架对同一算子的定义可能不完全一致。例如：

- PyTorch的`nn.Conv2d`的padding参数直接指定填充像素数
- TensorFlow的`tf.keras.layers.Conv2D`的padding参数是"valid"或"same"，语义不同

转换时需要正确处理这些语义差异，确保转换后模型的输出与原始模型一致。

**动态控制流**

PyTorch等框架支持动态控制流（如`if`语句、循环等），这些在转换为静态计算图时需要特殊处理。一些部署平台可能不支持动态控制流。

**自定义算子**

训练框架允许用户定义自定义算子（Custom Operator）。如果目标平台没有对应的实现，转换可能失败或需要额外适配工作。

**数值精度差异**

不同框架对默认数值精度可能有不同规定（如某些操作默认使用FP16 vs FP32）。转换时需要确保精度符合预期。

### 3.1.3 模型转换流程

典型的模型转换流程包括：

```
原始模型 → 格式解析 → 计算图构建 → 图优化 → 格式生成 → 目标模型
```

**1. 格式解析**

读取原始模型文件，解析其中的模型结构和参数。不同格式有不同的解析逻辑：

- 对于ONNX，使用protobuf定义的数据结构
- 对于TensorFlow SavedModel，使用SavedModel API
- 对于PyTorch，使用pickle加载state_dict

**2. 计算图构建**

将解析结果构建为推理引擎内部的计算图表示。计算图包含节点（算子）和边（张量），描述了各操作的依赖关系。

**3. 图优化**

对计算图进行各种优化：

- 删除训练相关的冗余操作
- 融合连续的计算操作
- 简化计算图结构
- 转换数据布局

**4. 格式生成**

将优化后的计算图转换为目标格式。这可能涉及序列化、编码等操作。

**5. 目标模型输出**

输出转换后的模型文件，以及可能的一些元数据（如输入输出形状、算子列表等）。

## 3.2 主流模型格式

### 3.2.1 ONNX

**ONNX（Open Neural Network Exchange）**是一个开放的模型格式标准，旨在实现不同AI框架之间的互操作性。ONNX由微软和Facebook在2017年联合推出，目前已成为模型转换的事实标准。

**ONNX的核心概念**

1. **计算图表示**：ONNX使用protobuf格式定义计算图，包含节点（NodeProto）、张量（TensorProto）等数据结构。

2. **标准化算子集**：ONNX定义了一套标准的算子（Operator），如卷积、池化、全连接等。每个算子有明确的输入输出语义。

3. **IR版本管理**：ONNX有明确的IR（Intermediate Representation）版本，不同版本可能有不同的算子定义或语义。

**ONNX算子示例**

ONNX中的一个卷积节点定义为：

```protobuf
node {
  input: "input"      // 输入张量名
  input: "weight"    // 权重张量名
  output: "output"   // 输出张量名
  name: "conv1"       // 节点名
  op_type: "Conv"      // 算子类型
  attribute {
    name: "kernel_shape"
    ints: [3, 3]
  }
  attribute {
    name: "strides"
    ints: [1, 1]
  }
  attribute {
    name: "pads"
    ints: [1, 1, 1, 1]
  }
}
```

**PyTorch转ONNX**

PyTorch提供内置的ONNX导出功能：

```python
import torch
import torch.onnx

# 定义模型
model = MyModel()
model.eval()

# 准备示例输入
dummy_input = torch.randn(1, 3, 224, 224)

# 导出为ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

**ONNX的使用场景**

- 框架间模型转换（如PyTorch→TensorRT）
- 模型部署到不同平台（如服务器、移动端）
- 模型性能分析工具（如Netron可视化）

### 3.2.2 TorchScript

**TorchScript**是PyTorch的序列化格式，可以将PyTorch模型导出为可以独立于Python环境运行的格式。TorchScript有两种主要形式：

1. **TorchScript代码**：通过`torch.jit.script`装饰器或函数将模型代码显式编译为TorchScript
2. **TorchScript追踪（Tracing）**：通过`torch.jit.trace`用示例输入"追踪"模型的执行路径生成TorchScript

**追踪模式（Tracing）**

```python
import torch

model = MyModel()
model.eval()

# 用示例输入追踪模型
traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))

# 保存追踪后的模型
traced_model.save('model_traced.pt')
```

追踪的优点是简单，不需要修改模型代码；缺点是只能捕获静态执行路径，动态控制流会"凝固"。

**脚本模式（Scripting）**

```python
import torch
import torch.nn as nn

@torch.jit.script
def my_function(x: torch.Tensor) -> torch.Tensor:
    if x.sum() > 0:
        return x * 2
    else:
        return x / 2

class MyModule(nn.Module):
    def forward(self, x):
        return my_function(x) + 1

model = MyModule()
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')
```

脚本模式支持完整的Python语法，包括动态控制流；缺点是有些Python语法可能不被支持。

**TorchScript的用途**

- 模型部署到生产环境
- 模型优化（如通过TorchScript进行图优化）
- 与C++ API集成（libtorch）

### 3.2.3 SafeTensors

**SafeTensors**是由Hugging Face推出的一种安全的模型序列化格式，专门用于大模型（LLM）的权重存储和传输。

**SafeTensors的设计目标**

1. **安全性**：不执行任意代码，避免pickle等格式的安全风险
2. **零拷贝加载**：mmap机制支持大模型的高效加载，无需将整个模型加载到内存
3. **快速保存**：可以只保存部分张量（如只保存更新部分）
4. **内存映射**：支持GPU直接通过PCIe访问

**SafeTensors格式**

SafeTensors文件是一个包含元数据和实际张量数据的结构：

```python
{
    "metadata": {
        "total_size": 135291469824,
        "format": "pt"
    },
    " tensors": {
        "model.embed_tokens.weight": {
            "dtype": "float16",
            "shape": [50257, 4096],
            "offset": 0,
            "data_file": "model.safetensors"
        },
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "float16",
            "shape": [4096, 4096],
            "offset": 412789760,
            "data_file": "model.safetensors"
        }
    }
}
```

**SafeTensors vs Pickle**

| 特性 | SafeTensors | Pickle |
|------|-------------|--------|
| 安全性 | 无代码执行风险 | 可能执行任意代码 |
| 加载速度 | mmap快速加载 | 需要反序列化 |
| 内存效率 | 支持零拷贝 | 全量加载到内存 |
| 适用场景 | 大模型权重存储 | 任意Python对象 |

### 3.2.4 其他模型格式

**TensorFlow SavedModel**

TensorFlow的推荐部署格式，包含计算图和权重：

```python
import tensorflow as tf

# 保存模型
model = tf.keras.models.load_model('my_model')
tf.saved_model.save(model, 'saved_model')

# 加载模型
loaded = tf.saved_model.load('saved_model')
```

SavedModel包含：
- `saved_model.pb`：计算图定义（protobuf格式）
- `variables/`：变量（权重）目录
- `assets/`：额外资源文件

**Caffe模型格式**

Caffe使用`*.caffemodel`存储权重，`deploy.prototxt`定义网络结构。虽然Caffe本身已较少使用，但其模型格式仍被一些推理引擎（如NCNN）支持。

**MindSpore MindIR**

华为MindSpore的中间表示格式，用于端云协同部署。

## 3.3 计算图基础

### 3.3.1 什么是计算图

**计算图（Computational Graph）**是表示数学运算的有向无环图（DAG）。在深度学习中，计算图描述了神经网络从输入到输出的完整计算流程。

**计算图的构成**

- **节点（Node/Operator）**：表示具体的数学运算，如卷积、矩阵乘法、ReLU等
- **边（Edge/Tensor）**：表示节点之间的数据流，即张量（多维数组）

**计算图示例**

对于神经网络`y = ReLU(Conv(x, W) + b)`：

```
x ──┐
    ├──→ Conv ──┐
W ──┘         ├──→ Add ──┐
               b ──────→──┘
                         │
                         ↓
                       ReLU
                         │
                         ↓
                         y
```

### 3.3.2 静态图与动态图

**静态计算图（Static Computational Graph）**

静态图在执行前就完全定义好了。代表框架：TensorFlow 1.x、ONNX。

特点：

- 构建和执行分开：先定义图，后执行
- 运行时开销小（图结构已知，可以高度优化）
- 对动态控制流支持有限

缺点：

- 调试困难（错误在图构建时而非运行时）
- Python语法受限（需要使用框架特定的API）

**动态计算图（Dynamic Computational Graph）**

动态图在执行过程中实时构建。代表框架：PyTorch、TensorFlow 2.x（默认）。

特点：

- 执行即构建：每个样本可能触发不同的图结构
- 调试友好（标准Python调试器可用）
- 自然支持动态控制流

缺点：

- 运行时开销（图构建开销分摊到每次执行）
- 优化空间有限（图结构在运行时才能确定）

**JIT编译的折中方案**

PyTorch的TorchScript和TensorFlow的XLA（Accelerated Linear Algebra）尝试结合两者优点：通过JIT编译在运行时生成优化的静态图。

### 3.3.3 计算图与自动求导

计算图的一个重要应用是**自动求导（Automatic Differentiation）**。通过构建计算图，可以自动计算梯度。

**前向传播 vs 反向传播**

- 前向传播：从输入计算输出
- 反向传播：从输出反向计算梯度

计算图使得梯度计算变得系统化和自动化。每个节点记录其前向操作和梯度函数，反向传播时自动链式求导。

**计算图在推理引擎中的简化**

推理时只需要前向传播，不需要反向传播。因此，推理引擎的计算图可以简化：

- 删除梯度计算相关的节点
- 合并连续操作（如Conv+BN+ReLU）
- 预计算常量表达式
- 优化内存布局

## 3.4 图优化技术

### 3.4.1 图优化的层次

图优化可以在多个层次进行：

**算子级别优化（Operator-level Optimization）**

单个算子的内部实现优化，如矩阵乘法的Strassen算法、卷积的Winograd算法等。这些优化通常在Kernel层实现。

**局部优化（Local Optimization）**

相邻算子组合的优化，如算子融合（Conv+BN→ConvBN）。这类优化在计算图层面进行。

**全局优化（Global Optimization）**

整个计算图的优化，如公共子图消除、死代码删除、常量折叠等。

### 3.4.2 算子融合

**算子融合（Operator Fusion）**是将多个连续的算子合并为一个复合算子的优化技术。

**为什么融合能提升性能？**

1. **减少内存访问**：中间结果不需要写回内存再读出
2. **减少Kernel调用开销**：一次Kernel执行替代多次
3. **增加计算密度**：更好地利用硬件的并行计算能力

**常见的融合模式**

**卷积+批量归一化+激活**

原始计算：`output = ReLU(BN(Conv(input, weight)))`

融合后等价于一个新的卷积操作，权重和偏置被融合：

```python
# 融合后的等效卷积
new_weight = gamma / sqrt(var + eps) * weight
new_bias = beta - gamma / sqrt(var + eps) * mean
output = Conv(input, new_weight) + new_bias
output = ReLU(output)
```

融合后的计算只需要一次卷积操作和一次ReLU操作。

**矩阵乘+偏置+激活**

```python
# 原始
output = Activation(MatMul(input, weight) + bias)

# 融合后
output = ActivationWithBias(input, weight, bias)
```

**多分支融合**

ResNet中的残差连接可以被融合为一个算子：

```python
# 原始
branch1 = conv1(input)
branch2 = conv2(input)
output = add(branch1, branch2)

# 融合后
output = residual_conv(input, conv1_weight, conv2_weight)
```

### 3.4.3 算子替换

**算子替换（Operator Replacement）**是用语义等价但更高效的算子替换原有算子的优化。

**常见的替换场景**

**1×1卷积替代全连接**

在全局池化后的分类层，FC层实际上是对每个通道做了加权求和，可以等价替换为1×1卷积：

```python
# 原始
x = global_avg_pool(features)  # [B, C, 1, 1]
x = x.view(B, C)               # [B, C]
output = fc(x, weight)         # [B, num_classes]

# 替换后
output = conv1x1(features, weight)  # [B, num_classes, 1, 1]
output = output.squeeze(-1).squeeze(-1)
```

**深度可分离卷积替代标准卷积**

Depthwise Separable Convolution是MobileNet的核心，可以大幅减少计算量：

```python
# 标准卷积
output = conv2d(input, weight)  # 计算量: H*W*C_in*C_out*K*K

# 深度可分离卷积
depthwise = conv2d(input, depthwise_weight)  # 逐通道卷积
pointwise = conv2d(depthwise, pointwise_weight)  # 1x1卷积
# 总计算量: H*W*C_in*K*K + H*W*C_in*C_out
```

### 3.4.4 布局转换

**布局转换（Layout Transformation）**是调整张量内存排列方式的优化。

**常见的内存布局**

- **NCHW**：Batch-Channels-Height-Width，GPU常用，通道在前
- **NHWC**：Batch-Height-Width-Channels，CPU/TensorFlow常用
- **NC4HW4/NCHW4**：通道拆分为4的倍数，适合SIMD向量化

**布局转换的时机**

1. **算子间布局对齐**：不同算子可能偏好不同的数据布局
2. **硬件特性适配**：GPU可能偏好NCHW，CPU可能偏好NHWC
3. **融合准备**：某些融合模式要求特定的输入布局

**转换的代价**

布局转换涉及数据拷贝，可能成为性能瓶颈。因此，优化策略应：

- 尽量减少不必要的布局转换
- 将转换融入融合算子中（如在Kernel中处理多种布局）
- 使用更高效的拷贝方法（如异步拷贝）

### 3.4.5 常量折叠

**常量折叠（Constant Folding）**是在编译时预先计算结果已知表达式的优化。

**示例**

```python
# 原始
x = constant(5)
y = constant(3)
z = x * y  # 在运行时计算

# 折叠后
z = constant(15)  # 编译时直接计算得到
```

在神经网络中，很多计算是常量（如图像归一化的均值、标准差），可以在转换阶段预先计算。

**常量折叠的益处**

- 减少运行时计算量
- 简化计算图，删除冗余节点
- 可能触发更多优化（如常量节点删除后，两个相邻算子可能可以融合）

### 3.4.6 冗余消除

**冗余消除（Redundancy Elimination）**删除不影响最终结果的计算。

**死代码删除（Dead Code Elimination）**

```python
# 原始
x = compute()
y = x + 1
z = y * 2
# x的结果未被使用（只有z被使用）

# 删除后
z = compute() + 1
z = z * 2
```

在推理时，梯度计算、训练相关的操作都是"死代码"。

**公共子图消除（Common Subexpression Elimination）**

```python
# 原始
a = x + y
b = x + y
c = a * 2

# 消除后
a = x + y
b = a  # 复用a的结果
c = a * 2
```

## 3.5 中间表示

### 3.5.1 什么是中间表示

**中间表示（Intermediate Representation，IR）**是模型在不同格式之间转换时的桥梁。IR定义了模型的结构化描述方式，使得转换逻辑可以统一处理。

**为什么需要IR？**

假设有N种模型格式，如果两两之间直接转换，需要N×(N-1)种转换器。但如果都先转换为统一的IR，再从IR转换到目标格式，只需要2N种转换器。

```
格式A ──┐
        ├──→ IR ──┬──→ 格式X
格式B ──┘         │
        └──→ 格式Y
```

**IR的设计要求**

1. **表达能力**：能够描述各种神经网络的结构和操作
2. **可优化性**：便于进行各种图优化
3. **可序列化**：能够保存和加载
4. **跨平台**：不依赖特定框架或硬件

### 3.5.2 常见IR格式

**ONNX作为IR**

ONNX本身就可以作为IR使用。很多推理引擎将ONNX作为导入格式，然后在内部进行进一步优化。

优点：

- 生态完善，工具链成熟
- 很多框架直接支持导出ONNX

缺点：

- ONNX算子集有限，可能无法表达某些自定义操作
- 版本兼容性有时是问题

**厂商自定义IR**

大型AI公司通常有自己的IR格式：

- TensorRT的IGPU/Plan格式
- OpenVINO的IR格式（XML+Bin）
- MNN的自定义格式
- MindSpore的MindIR

这些IR通常针对自家硬件优化，支持更多专属算子。

### 3.5.3 IR的组成部分

典型的IR包含以下组成部分：

**模型元信息**

```protobuf
message ModelProto {
    string name = 1;           // 模型名称
    int64 ir_version = 2;       // IR版本
    repeated string producer_name = 3;
    repeated string producer_version = 4;
    repeated string inputs = 5;   // 输入张量信息
    repeated string outputs = 6;   // 输出张量信息
    repeated OperatorSetIdProto opset_import = 7;
}
```

**计算图**

```protobuf
message GraphProto {
    repeated NodeProto node = 1;  // 节点列表
    repeated TensorProto initializer = 2;  // 权重常量
    string name = 3;
}
```

**节点定义**

```protobuf
message NodeProto {
    string name = 1;           // 节点名称
    string op_type = 2;         // 算子类型
    repeated string input = 3;  // 输入张量
    repeated string output = 4;// 输出张量
    map<string, AttributeProto> attribute = 5;  // 算子属性
}
```

**张量定义**

```protobuf
message TensorProto {
    repeated int64 dims = 1;     // 形状
    DataType data_type = 2;    // 数据类型
    repeated float float_data = 3;  // 权重数据
    string name = 4;
}
```

### 3.5.4 基于IR的优化流程

```
导入模型 → 解析为IR → 图优化 → IR优化后 → 生成目标格式
```

**1. 导入模型**

解析各种格式的模型文件，转换为统一的IR表示。这个过程需要处理：

- 框架特定的API到标准算子的映射
- 非标准操作的分解或适配
- 模型版本的兼容处理

**2. 图级别优化**

在IR上进行与硬件无关的优化：

- 常量折叠
- 死代码删除
- 公共子图消除
- 算子简化

**3. 硬件适配优化**

针对目标硬件的优化：

- 布局转换
- 算子融合
- Kernel选择

**4. 代码生成**

将优化后的IR生成为目标格式：

- 序列化为protobuf（如ONNX）
- 生成硬件相关的执行计划（如TensorRT Engine）
- 生成特定平台的代码（如移动端的C++代码）

## 3.6 模型转换实践

### 3.6.1 PyTorch转ONNX实践

**基本转换步骤**

```python
import torch
import torch.onnx
import torchvision.models as models

# 1. 加载预训练模型
model = models.resnet50(pretrained=True)
model.eval()

# 2. 准备示例输入
dummy_input = torch.randn(1, 3, 224, 224)

# 3. 执行转换
torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# 4. 验证模型
import onnx
model_onnx = onnx.load("resnet50.onnx")
onnx.checker.check_model(model_onnx)
print("ONNX模型验证通过！")
```

**常见问题和解决方案**

1. **动态shape支持**

```python
# 使用dynamic_axes参数
torch.onnx.export(
    model, dummy_input, "model.onnx",
    dynamic_axes={
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size'}
    }
)
```

2. **算子不支持**

某些PyTorch算子在ONNX中没有直接对应，需要自定义：

```python
@torch.onnx.symbolic_helper.parse_args('v', 'v', 'i')
def my_custom_op(g, input1, input2, alpha):
    return g.op("CustomOp", input1, input2, alpha_s=alpha)

# 注册自定义算子
torch.onnx.register_custom_op_symbolic('::my_custom_op', my_custom_op, opset_version=11)
```

### 3.6.2 ONNX转TensorRT实践

```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

# 1. 创建Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 2. 创建Builder
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB

# 3. 解析ONNX模型
parser = trt.OnnxParser(network, TRT_LOGGER)
with open("resnet50.onnx", "rb") as f:
    parser.parse(f.read())

# 4. 配置优化
builder.max_batch_size = 32
config.set_flag(trt.BuilderFlag.FP16)  # 启用FP16优化

# 5. 生成Engine
engine = builder.build_serialized_network(network, config)

# 6. 保存Engine
with open("resnet50.trt", "wb") as f:
    f.write(engine)

print("TensorRT Engine生成成功！")
```

**TensorRT优化策略**

- **FP16/INT8量化**：减少计算精度需求，加速推理
- **张量融合**：减少内存访问
- **内核自动调优**：为特定输入shape选择最优算法
- **内存优化**：复用中间结果内存

### 3.6.3 转换工具链

**MMDeploy**

OpenMMLab的模型部署工具，支持PyTorch→ONNX→推理引擎的完整流程：

```bash
# 安装
pip install mmdeploy

# 转换模型
python tools/deploy.py \
    configs/mmdetection/onnx-sdk/onnx2sdk.py \
    mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn.py \
    mmdetection/checkpoints/faster-rcnn_r50_fpn_1x_coco_20200130-3c540c11.pth \
    --deploy-cfg configs/mmdetection/onnx-sdk/onnx2sdk.py
```

**TVM Relay**

TVM的图IR，支持从多种框架导入模型并进行优化：

```python
import tvm
from tvm import relay

# 从ONNX导入
mod, params = relay.frontend.from_onnx(onnx_model)

# 应用优化
with tvm.transform.PassContext(opt_level=3):
    mod = relay.optimize(mod, target="llvm")

# 编译
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="llvm", params=params)
```

## 本章小结

1. **模型转换必要性**：训练框架多样性、部署环境差异、优化需求和跨平台支持都需要模型转换。转换面临算子语义差异、动态控制流、自定义算子等挑战。

2. **主流模型格式**：ONNX是开放的互操作性标准；TorchScript是PyTorch的序列化格式支持追踪和脚本两种模式；SafeTensors是Hugging Face推出的安全高效的大模型格式。

3. **计算图基础**：计算图是表示数学运算的有向无环图，由节点（算子）和边（张量）构成。分为静态图（先定义后执行）和动态图（执行即构建）两种。

4. **图优化技术**：算子融合减少内存访问和Kernel调用；算子替换用高效算子替代低效算子；布局转换适配硬件；常量折叠预计算常量；冗余消除删除无用计算。

5. **中间表示IR**：IR是模型转换的桥梁，使N种格式两两转换变成2N种转换器。ONNX本身即可作为IR使用，大厂商也有自己的IR格式。

## 思考与练习

1. **概念理解**：解释为什么模型转换需要"中间表示"这个概念？如果没有IR，直接在任意两种格式之间转换会有什么问题？

2. **原理分析**：ONNX的算子集是标准化的，但不同推理引擎对ONNX的支持程度不同。请分析可能的原因，以及在实际部署中应该如何处理这种不一致性。

3. **设计思考**：假设你需要为一个新的AI芯片开发模型转换工具，该芯片有自己的模型格式和算子定义。请说明你会如何设计转换工具的架构，需要考虑哪些关键问题？

4. **实践应用**：对比TorchScript的追踪模式（trace）和脚本模式（script）的优缺点。什么场景下应该选择哪种模式？

5. **优化分析**：分析算子融合为什么能够提升推理性能。如果融合Conv+BN+ReLU为一个算子，可能面临哪些技术挑战？

6. **扩展调研**：调研SafeTensors格式相比传统的pickle/h5格式在LLM场景下的优势，并分析其实现原理（内存映射、零拷贝加载等）。
