# PyTorch基础

## 学习目标

- 理解PyTorch的核心数据结构：Tensor（张量）
- 掌握PyTorch Tensor的创建、操作和基本运算
- 理解PyTorch的自动求导机制（autograd）
- 学会使用PyTorch构建简单的神经网络
- 掌握PyTorch的数据加载和训练流程
- 了解模型的保存与加载方法

## 3.1 PyTorch概述

PyTorch是Facebook开发的开源深度学习框架，以其动态计算图（Dynamic Computation Graph）和Python优先的设计理念，成为当前最流行的深度学习框架之一。

### 3.1.1 安装与导入

```python
# 安装PyTorch（CPU版本）
# pip install torch

# 安装PyTorch（CUDA版本，根据CUDA版本选择）
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# 导入PyTorch
import torch
```

### 3.1.2 Tensor概述

Tensor（张量）是PyTorch的核心数据结构，可以理解为多维数组，与NumPy的ndarray非常相似，但Tensor支持GPU加速计算和自动求导。

```python
# 标量（0维张量）
scalar = torch.tensor(3.14)
print(scalar)        # tensor(3.1400)
print(scalar.dim())  # 0

# 向量（1维张量）
vector = torch.tensor([1, 2, 3, 4, 5])
print(vector)        # tensor([1, 2, 3, 4, 5])
print(vector.shape)  # torch.Size([5])

# 矩阵（2维张量）
matrix = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
print(matrix)
# tensor([[1, 2, 3],
#         [4, 5, 6]])
print(matrix.shape)  # torch.Size([2, 3])

# 3维张量
tensor_3d = torch.randn(2, 3, 4)  # 随机初始化的3维张量
print(tensor_3d.shape)  # torch.Size([2, 3, 4])
```

## 3.2 Tensor创建

PyTorch提供了多种创建Tensor的方式。

### 3.2.1 从数据创建

```python
# 从Python列表创建
a = torch.tensor([1, 2, 3, 4, 5])
print(a)  # tensor([1, 2, 3, 4, 5])

# 从NumPy数组创建
import numpy as np
np_array = np.array([[1, 2, 3], [4, 5, 6]])
torch_tensor = torch.from_numpy(np_array)
print(torch_tensor)
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# Tensor转NumPy
back_to_np = torch_tensor.numpy()
print(back_to_np)
# [[1 2 3]
#  [4 5 6]]

# 注意：共享内存（修改一方会影响另一方）
torch_tensor[0, 0] = 99
print(np_array[0, 0])  # 99

# 不共享内存的转换
copy_tensor = torch_tensor.clone().numpy()
```

### 3.2.2 预定义Tensor

```python
# 全0Tensor
zeros_1d = torch.zeros(5)
zeros_2d = torch.zeros(3, 4)
zeros_3d = torch.zeros(2, 3, 4)
print(zeros_2d)
# tensor([[0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])

# 全1Tensor
ones = torch.ones(2, 3)
print(ones)
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])

# 填充指定值
full = torch.full((2, 3), 7.0)
print(full)
# tensor([[7., 7., 7.],
#         [7., 7., 7.]])

# 单位矩阵
eye = torch.eye(4)
print(eye)
# tensor([[1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 0., 0., 1.]])
```

### 3.2.3 范围和序列

```python
# arange：类似Python的range
a = torch.arange(10)          # 0到9
print(a)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

b = torch.arange(1, 10)       # 1到9
print(b)  # tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

c = torch.arange(0, 10, 2)    # 0到9，步长2
print(c)  # tensor([0, 2, 4, 6, 8])

# linspace：等差数列
d = torch.linspace(0, 1, 5)   # 0到1之间均匀取5个点
print(d)  # tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

### 3.2.4 随机Tensor

```python
# 设置随机种子（结果可复现）
torch.manual_seed(42)

# 均匀分布[0, 1)
uniform = torch.rand(3, 4)
print(uniform)
# tensor([[0.3745, 0.9507, 0.7320, 0.5987],
#         [0.1560, 0.1560, 0.0581, 0.8662],
#         [0.6011, 0.7081, 0.0206, 0.9699]])

# 标准正态分布（均值0，方差1）
normal = torch.randn(3, 4)
print(normal)

# 整数随机Tensor
integers = torch.randint(0, 10, (3, 4))  # 0到9
print(integers)

# 随机打乱
arr = torch.arange(10)
shuffled = arr[torch.randperm(len(arr))]  # 随机排列索引
print(shuffled)

# 生成随机初始化的Tensor，指定尺寸
linear = torch.randn(512, 1024)  # 权重初始化常用
```

## 3.3 Tensor属性

```python
x = torch.randn(3, 4, 5)

# shape：形状
print(x.shape)       # torch.Size([3, 4, 5])
print(x.size())      # torch.Size([3, 4, 5])

# ndim/dim：维度数
print(x.dim())       # 3

# dtype：数据类型
print(x.dtype)       # torch.float32

# device：所在设备
print(x.device)      # cpu

# numel：元素总数
print(x.numel())     # 60

# layout：内存布局
print(x.layout)      # torch.strided
```

**Tensor数据类型**：

| dtype | 说明 |
|-------|------|
| torch.float32 / torch.float | 32位浮点 |
| torch.float64 / torch.double | 64位浮点 |
| torch.float16 / torch.half | 16位浮点 |
| torch.int8 | 8位整数 |
| torch.int16 / torch.short | 16位整数 |
| torch.int32 / torch.int | 32位整数 |
| torch.int64 / torch.long | 64位整数 |
| torch.bool | 布尔 |

```python
# 指定数据类型
a = torch.tensor([1, 2, 3], dtype=torch.float32)
print(a.dtype)  # torch.float32

# 类型转换
b = a.to(torch.int32)
print(b.dtype)  # torch.int32

c = a.int()     # 简写形式
print(c.dtype)  # torch.int32

# 与NumPy的类型对应
print(torch.float32 == np.float32)  # True
print(torch.int64 == np.int64)       # True
```

## 3.4 Tensor操作

### 3.4.1 索引与切片

PyTorch的索引和切片语法与NumPy非常相似。

```python
x = torch.arange(12).reshape(3, 4)
print(x)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# 基本索引
print(x[0])       # tensor([0, 1, 2, 3])
print(x[1, 2])    # tensor(6)
print(x[-1, -1])  # tensor(11)

# 切片
print(x[:, 0])    # tensor([0, 4, 8])（第一列）
print(x[1:, :2])  # tensor([[4, 5], [8, 9]]）（后两行的前两列）
print(x[::2, ::2])  # tensor([[0, 2], [8, 10]])（跳行跳列）

# 布尔索引
mask = x > 5
print(mask)
# tensor([[False, False, False, False],
#         [False, False,  True,  True],
#         [ True,  True,  True,  True]])
print(x[mask])  # tensor([ 6,  7,  8,  9, 10, 11])

# 条件索引
print(x[x > 5])  # tensor([ 6,  7,  8,  9, 10, 11])
```

### 3.4.2 形状操作

```python
# reshape：改变形状
x = torch.arange(12)
y = x.reshape(3, 4)
print(y.shape)  # torch.Size([3, 4])

# view：类似reshape，但不复制数据
z = x.view(3, 4)

# flatten：展平为一维
flat = y.flatten()
print(flat.shape)  # torch.Size([12])

# squeeze/unsqueeze：移除/添加单维度
a = torch.randn(1, 3, 1, 4)
print(a.shape)  # torch.Size([1, 3, 1, 4])

b = torch.squeeze(a)  # 移除所有单维度
print(b.shape)  # torch.Size([3, 4])

c = torch.squeeze(a, dim=0)  # 移除指定维度的单维度
print(c.shape)  # torch.Size([3, 1, 4])

d = torch.unsqueeze(b, dim=0)  # 添加单维度
print(d.shape)  # torch.Size([1, 3, 4])

# transpose：转置（2维）
x = torch.randn(3, 4)
y = x.T  # 或 x.transpose(0, 1)
print(y.shape)  # torch.Size([4, 3])

# permute：多维转置
x = torch.randn(2, 3, 4, 5)
y = x.permute(2, 0, 3, 1)  # (2,3,4,5) -> (4,2,5,3)
print(y.shape)  # torch.Size([4, 2, 5, 3])

# repeat：重复Tensor
x = torch.tensor([1, 2, 3])
y = x.repeat(3)  # 重复3次
print(y)  # tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])

z = x.repeat(2, 1)  # 沿第0维重复2次，第1维重复1次
print(z)
# tensor([[1, 2, 3],
#         [1, 2, 3]])
```

### 3.4.3 连接与分割

```python
# cat：沿现有轴连接
a = torch.randn(2, 3)
b = torch.randn(2, 3)

c = torch.cat([a, b], dim=0)  # 沿第0维连接
print(c.shape)  # torch.Size([4, 3])

d = torch.cat([a, b], dim=1)  # 沿第1维连接
print(d.shape)  # torch.Size([2, 6])

# stack：沿新轴连接（会创建新维度）
e = torch.stack([a, b], dim=0)
print(e.shape)  # torch.Size([2, 2, 3])

# split：分割
x = torch.randn(8, 4)
# 按大小分割
splits = torch.split(x, [2, 3, 3], dim=0)  # 分割为2,3,3

# chunk：均匀分割
chunks = torch.chunk(x, 4, dim=0)  # 沿第0维均匀分割为4份

# split与chunk对比
print([s.shape for s in splits])  # [torch.Size([2, 4]), torch.Size([3, 4]), torch.Size([3, 4])]
print([c.shape for c in chunks])    # [torch.Size([2, 4]), torch.Size([2, 4]), torch.Size([2, 4]), torch.Size([2, 4])]
```

## 3.5 Tensor运算

### 3.5.1 逐元素运算

```python
a = torch.tensor([1.0, 2.0, 3.0, 4.0])

# 基本算术运算
print(a + 1)     # tensor([2., 3., 4., 5.])
print(a - 1)     # tensor([0., 1., 2., 3.])
print(a * 2)     # tensor([2., 4., 6., 8.])
print(a / 2)     # tensor([0.5000, 1.0000, 1.5000, 2.0000])
print(a ** 2)    # tensor([ 1.,  4.,  9., 16.])

# 逐元素比较
b = torch.tensor([4, 2, 2, 5])
print(a > b)     # tensor([False, False,  True, False])
print(a == b)    # tensor([False,  True, False, False])

# 常用数学函数
x = torch.tensor([0, 30, 60, 90]) * np.pi / 180
print(torch.sin(x))  # 正弦
print(torch.cos(x))  # 余弦
print(torch.exp(x))  # 指数
print(torch.log(x))  # 自然对数
print(torch.sqrt(x)) # 平方根
```

### 3.5.2 矩阵运算

```python
# 矩阵乘法
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.mm(A, B)  # 或 A @ B
print(C.shape)  # torch.Size([3, 5])

# 批量矩阵乘法
A = torch.randn(10, 3, 4)
B = torch.randn(10, 4, 5)
C = torch.bmm(A, B)  # 批量矩阵乘法
print(C.shape)  # torch.Size([10, 3, 5])

# 更通用的批矩阵乘法
C = torch.matmul(A, B)  # 支持广播

# 向量点积
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])
dot = torch.dot(v1, v2)
print(dot)  # tensor(32.)

# 求范数
x = torch.tensor([3.0, 4.0])
l2_norm = torch.norm(x)  # 默认L2范数
print(l2_norm)  # tensor(5.)

l1_norm = torch.norm(x, p=1)  # L1范数
print(l1_norm)  # tensor(7.)

# 矩阵转置
A = torch.randn(3, 4)
print(A.T.shape)  # torch.Size([4, 3])
```

### 3.5.3 归约操作

```python
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

# 求和
print(torch.sum(a))           # tensor(21)
print(torch.sum(a, dim=0))    # tensor([5, 7, 9])（按列）
print(torch.sum(a, dim=1))    # tensor([ 6, 15])（按行）

# 均值
print(torch.mean(a))          # tensor(3.5000)
print(torch.mean(a, dim=0))   # tensor([2.5000, 3.5000, 4.5000])

# 标准差
print(torch.std(a))           # tensor(1.7078)

# 最大最小值
print(torch.min(a))           # tensor(1)
print(torch.max(a))           # tensor(6)
print(torch.argmin(a))        # tensor(0)（索引）
print(torch.argmax(a))        # tensor(5)

# 沿指定维度求最值
print(torch.max(a, dim=0))     # 返回(values, indices)
values, indices = torch.max(a, dim=1)
print(values)   # tensor([3, 6])
print(indices)  # tensor([2, 2])

# 累积操作
print(torch.cumsum(a, dim=0))
# tensor([[1, 2, 3],
#         [5, 7, 9]])
```

### 3.5.4 广播机制

PyTorch的广播机制与NumPy类似，但需要注意维度匹配。

```python
# 基本广播
a = torch.randn(3, 4)
b = torch.randn(4)
c = a + b  # b被广播到(3,4)

# 显式添加维度
a = torch.randn(3, 4)
b = torch.randn(3, 1)  # (3,1)广播到(3,4)
c = a + b

# 批量处理示例
images = torch.randn(32, 3, 224, 224)  # batch_size=32, 3通道, 224x224
mean = torch.mean(images, dim=[2, 3])  # (32, 3)
# mean需要reshape为(32, 3, 1, 1)才能与images广播
mean = mean.reshape(32, 3, 1, 1)
normalized = images - mean
print(normalized.shape)  # (32, 3, 224, 224)
```

## 3.6 自动求导（autograd）

自动求导（Automatic Differentiation）是PyTorch的核心特性之一，能够自动计算梯度。

### 3.6.1 requires_grad

默认情况下，Tensor不会追踪计算历史。设置 `requires_grad=True` 开启梯度追踪。

```python
# 创建需要求导的Tensor
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x.requires_grad)  # True

# 简写方式
x = torch.ones(3, requires_grad=True)

# 从不需要求导的Tensor创建时，可显式指定
y = torch.tensor([1.0, 2.0, 3.0])
z = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = y + z  # w.requires_grad 为 True
```

### 3.6.2 计算梯度

```python
# 示例：计算 y = x^2 在 x = [1, 2, 3] 处的梯度
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# y = [1, 4, 9]
# dy/dx = 2x = [2, 4, 6]

# backward()自动计算梯度
y.sum().backward()  # 对标量调用，或对非标量y.sum().backward()

print(x.grad)  # tensor([2., 4., 6.])
```

**非标量输出的反向传播**：

```python
# 非标量Tensor需要指定gradient参数或先求和
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # y = [1, 4, 9]

# 方式1：求和后反向传播
y.sum().backward()

# 方式2：传入与y形状相同的gradient参数
# y.backward(torch.ones_like(y))  # 效果等同于y.sum().backward()

print(x.grad)  # tensor([2., 4., 6.])
```

### 3.6.3 计算图

PyTorch使用动态计算图（Dynamic Computation Graph），每次运算都会构建新的计算图。

```python
# 计算图示例
x = torch.tensor([1.0], requires_grad=True)
y = x + 2       # 创建加法节点
z = y ** 2      # 创建幂运算节点
w = z * 3       # 创建乘法节点

w.backward()
print(x.grad)  # dy/dx = ?

# 手动验证：w = 3*(x+2)^2
# dw/dx = 6*(x+2) = 6*3 = 18
print(x.grad)  # tensor([18.])
```

**叶子节点与依赖节点**：

```python
x = torch.tensor([1.0], requires_grad=True)
y = x + 2
z = y ** 2

print(x.is_leaf)   # True（用户创建的）
print(y.is_leaf)   # False（由运算生成）
print(z.is_leaf)   # False

print(x.grad_fn)   # None（叶子节点没有grad_fn）
print(y.grad_fn)   # <AddBackward0>
print(z.grad_fn)   # <PowBackward0>
```

### 3.6.4 梯度累积与 detach

计算图会累积梯度，多次反向传播时梯度会累加。

```python
# 梯度累积示例
x = torch.tensor([1.0], requires_grad=True)

# 第一次计算
y = x ** 2
y.backward()
print(x.grad)  # tensor([2.])

# 第二次计算前需要清零
x.grad.zero_()

# 第二次计算
z = x ** 3
z.backward()
print(x.grad)  # tensor([3.])

# 如果不清零，梯度会累加：2 + 3 = 5
```

**detach()：分离计算图**：

```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2

# 分离y，使z脱离计算图
z = y.detach()
print(z.requires_grad)  # False

# 或者创建不需要梯度的副本
y_no_grad = y.detach().requires_grad_(False)
```

### 3.6.5 with torch.no_grad()

在推理（inference）阶段，不需要梯度追踪，使用 `no_grad()` 可以节省内存和计算。

```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2

# 不追踪梯度
with torch.no_grad():
    z = y + 1
    print(z.requires_grad)  # False

# 装饰器形式
@torch.no_grad()
def inference(model, x):
    return model(x)

# 或者全局设置
torch.set_grad_enabled(False)
# ... 推理代码 ...
torch.set_grad_enabled(True)
```

## 3.7 神经网络构建

PyTorch通过 `torch.nn` 模块提供了构建神经网络的工具。

### 3.7.1 常用层

```python
import torch.nn as nn

# Linear（全连接层）
linear = nn.Linear(in_features=10, out_features=5, bias=True)
input_tensor = torch.randn(3, 10)  # batch_size=3, 输入特征=10
output = linear(input_tensor)
print(output.shape)  # torch.Size([3, 5])

# 权重和偏置
print(linear.weight.shape)  # torch.Size([5, 10])
print(linear.bias.shape)    # torch.Size([5])

# Conv2d（卷积层）
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
# 输入：(batch, channels, height, width)
input_tensor = torch.randn(1, 3, 32, 32)
output = conv(input_tensor)
print(output.shape)  # torch.Size([1, 16, 32, 32])

# MaxPool2d（最大池化）
pool = nn.MaxPool2d(kernel_size=2, stride=2)
output = pool(output)
print(output.shape)  # torch.Size([1, 16, 16, 16])

# AvgPool2d（平均池化）
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
output = avg_pool(output)
print(output.shape)  # torch.Size([1, 16, 8, 8])
```

### 3.7.2 激活函数

```python
import torch.nn.functional as F

x = torch.randn(5)

# ReLU
relu = F.relu(x)
print(relu)  # max(0, x)

# Sigmoid
sigmoid = torch.sigmoid(x)
print(sigmoid)  # 1 / (1 + exp(-x))

# Tanh
tanh = torch.tanh(x)
print(tanh)

# Softmax（沿指定维度）
x = torch.randn(2, 5)
softmax = F.softmax(x, dim=1)  # 沿第1维归一化
print(softmax)  # 每行和为1
```

### 3.7.3 构建模型

PyTorch有两种构建模型的方式：Sequential和类继承。

**方式1：Sequential**

```python
import torch.nn as nn

# 简单模型用Sequential
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 测试模型
x = torch.randn(32, 784)  # batch_size=32
output = model(x)
print(output.shape)  # torch.Size([32, 10])
```

**方式2：类继承**

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNet()
print(model)
# SimpleNet(
#   (fc1): Linear(in_features=784, out_features=256, bias=True)
#   (fc2): Linear(in_features=256, out_features=128, bias=True)
#   (fc3): Linear(in_features=128, out_features=10, bias=True)
#   (relu): ReLU()
# )

# 访问参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

### 3.7.4 损失函数

```python
import torch.nn as nn

# 交叉熵损失（用于分类）
criterion = nn.CrossEntropyLoss()

# 模拟预测和目标
outputs = torch.randn(4, 10)  # 4个样本，10个类别
targets = torch.tensor([0, 5, 3, 7])  # 真实类别索引

loss = criterion(outputs, targets)
print(f"CrossEntropyLoss: {loss.item():.4f}")

# MSE损失（用于回归）
mse_loss = nn.MSELoss()
predictions = torch.tensor([[1.0], [2.0], [3.0]])
targets = torch.tensor([[1.5], [2.5], [2.5]])
loss = mse_loss(predictions, targets)
print(f"MSELoss: {loss.item():.4f}")

# BCE损失（二分类）
bce_loss = nn.BCELoss()
sigmoid = nn.Sigmoid()
pred = sigmoid(torch.randn(4, 1))
target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
loss = bce_loss(pred, target)
print(f"BCELoss: {loss.item():.4f}")
```

### 3.7.5 优化器

```python
import torch.optim as optim

# 创建模型
model = SimpleNet()

# SGD优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam优化器（更常用）
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 典型训练步骤
for data, target in dataloader:
    optimizer.zero_grad()    # 清零梯度
    output = model(data)    # 前向传播
    loss = criterion(output, target)  # 计算损失
    loss.backward()         # 反向传播
    optimizer.step()        # 更新参数
```

## 3.8 数据加载

### 3.8.1 Dataset与DataLoader

```python
from torch.utils.data import Dataset, DataLoader

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建数据集
X = torch.randn(1000, 784)  # 1000个样本
y = torch.randint(0, 10, (1000,))  # 10个类别
dataset = CustomDataset(X, y)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# 迭代数据
for batch_data, batch_labels in dataloader:
    print(batch_data.shape, batch_labels.shape)
    # torch.Size([32, 784]) torch.Size([32])
    break
```

### 3.8.2 torchvision数据集

PyTorch提供了常用的计算机视觉数据集。

```python
import torchvision
import torchvision.transforms as transforms

# 下载并加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

## 3.9 训练流程

完整的PyTorch训练流程如下：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 准备数据
X = torch.randn(1000, 20)
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. 定义模型
class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = RegressionNet()
print(model)

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    running_loss = 0.0

    for batch_X, batch_y in dataloader:
        # 前向传播
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.item()

    # 打印epoch平均损失
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 5. 评估模型
model.eval()  # 设置为评估模式
with torch.no_grad():
    test_X = torch.randn(100, 20)
    test_y = torch.randn(100, 1)
    predictions = model(test_X)
    test_loss = criterion(predictions, test_y)
    print(f"Test Loss: {test_loss.item():.4f}")
```

## 3.10 模型保存与加载

### 3.10.1 保存模型

```python
# 保存整个模型（不推荐，包含整个模型结构）
torch.save(model, 'model.pth')

# 保存模型参数（推荐，轻量且灵活）
torch.save(model.state_dict(), 'model_weights.pth')

# 保存检查点（保存训练状态）
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')
```

### 3.10.2 加载模型

```python
# 加载整个模型
loaded_model = torch.load('model.pth')

# 加载模型参数
model = RegressionNet()  # 先创建模型结构
model.load_state_dict(torch.load('model_weights.pth'))

# 加载检查点
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### 3.10.3 GPU训练

```python
# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 将模型和数据移动到GPU
model = model.to(device)

# 训练循环中
for batch_X, batch_y in dataloader:
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)

    predictions = model(batch_X)
    loss = criterion(predictions, batch_y)
    # ...

# 从GPU移回CPU
model = model.cpu()
predictions = predictions.cpu()
```

## 本章小结

本章介绍了PyTorch的基础知识，主要包括：

1. **Tensor创建**：从数据创建、预定义Tensor、范围数组、随机Tensor，以及与NumPy的相互转换。

2. **Tensor属性**：shape、dtype、device、numel等属性，以及数据类型转换。

3. **Tensor操作**：索引与切片、形状操作（reshape、view、transpose、permute等）、连接与分割。

4. **Tensor运算**：逐元素运算、矩阵运算、归约操作、广播机制。

5. **自动求导（autograd）**：requires_grad、backward()、计算图、梯度累积、no_grad上下文管理器。

6. **神经网络构建**：常用层（Linear、Conv2d、Pool）、激活函数、模型定义方式（Sequential和类继承）、损失函数、优化器。

7. **数据加载**：Dataset、DataLoader、torchvision数据集。

8. **训练流程**：完整的前向传播、反向传播、参数更新循环。

9. **模型保存与加载**：state_dict方式、完整模型、检查点保存、GPU/CPU迁移。

PyTorch的动态计算图和Python化API使其成为深度学习研究和实践的首选框架。本章内容为后续构建更复杂的神经网络模型奠定了基础。

## 思考与练习

1. 创建两个形状为 (3, 4) 的随机Tensor A和B，实现：
   - 矩阵乘法 A @ B^T
   - 沿第0维计算元素-wise乘积（使用广播）
   - 计算每行的L2范数

2. 实现一个简单的多层感知机（MLP），包含两个隐藏层（每层128个神经元），用于对MNIST手写数字进行分类（10类）。使用ReLU激活函数，输出层使用LogSoftmax。

3. 给定一个自定义数据集类 `ImageDataset`，包含图像路径和标签，编写数据增强的transforms pipeline，包括：随机水平翻转、随机旋转（-15到15度）、归一化到[-1,1]。

4. 使用PyTorch实现一个学习率调度器（Learning Rate Scheduler），在训练过程中根据epoch数动态调整学习率：
   - 前5个epoch使用固定学习率0.01
   - 之后每个epoch将学习率衰减为原来的一半

5. 实现一个自定义损失函数 `FocalLoss`，用于处理类别不平衡的分类问题。Focal Loss的公式为：
   - FL(p_t) = -α_t(1 - p_t)^γ log(p_t)
   - 其中 p_t 是模型预测的正确类别的概率，α 和 γ 是超参数。
