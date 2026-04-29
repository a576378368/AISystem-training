# NumPy基础

## 学习目标

- 理解NumPy的核心概念：ndarray数组
- 掌握NumPy数组的创建、索引和切片操作
- 熟悉NumPy的形状操作（reshape、transpose等）
- 掌握NumPy的逐元素运算和矩阵运算
- 理解NumPy的广播机制及其应用场景
- 了解常用NumPy函数和性能注意事项

## 2.1 NumPy概述

NumPy（Numerical Python）是Python生态中用于数值计算的基础库，提供了高性能的多维数组对象 `ndarray` 以及丰富的数学函数库。几乎所有深度学习框架（PyTorch、TensorFlow等）的张量操作都借鉴了NumPy的设计思想。

### 2.1.1 安装与导入

```python
# 安装NumPy
# pip install numpy

# 导入NumPy（约定使用np作为别名）
import numpy as np
```

### 2.1.2 NumPy的核心：ndarray

`ndarray`（N-dimensional array）是NumPy的核心数据结构，是一个多维同质数组（即所有元素类型相同）。

```python
# 一维数组
a = np.array([1, 2, 3, 4, 5])
print(a)        # [1 2 3 4 5]
print(type(a)) # <class 'numpy.ndarray'>

# 二维数组（矩阵）
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(b)
# [[1 2 3]
#  [4 5 6]]

# 三维数组
c = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])
print(c)
# [[[1 2]
#   [3 4]]
#  [[5 6]
#   [7 8]]]
```

**NumPy数组 vs Python列表**：

| 特性 | Python列表 | NumPy数组 |
|------|-----------|-----------|
| 元素类型 | 任意类型 | 相同类型 |
| 内存占用 | 较大 | 紧凑 |
| 计算速度 | 慢（逐元素操作） | 快（向量化） |
| 功能 | 基础列表操作 | 丰富数学函数 |

```python
# Python列表：每个元素都是独立对象
py_list = [1, 2, 3, 4, 5]
print(py_list * 2)  # [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]（重复连接）

# NumPy数组：逐元素乘法
np_array = np.array([1, 2, 3, 4, 5])
print(np_array * 2)  # [2, 4, 6, 8, 10]（数学运算）
```

## 2.2 创建数组

NumPy提供了多种创建数组的方式。

### 2.2.1 从Python列表转换

```python
# 从列表创建
list1 = [1, 2, 3, 4, 5]
arr1 = np.array(list1)

# 指定数据类型
arr2 = np.array([1, 2, 3], dtype=np.float32)
print(arr2.dtype)  # float32

# 嵌套列表创建多维数组
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(matrix.shape)  # (2, 3)
```

### 2.2.2 预定义数组

```python
# 全0数组
zeros_1d = np.zeros(5)           # 一维
zeros_2d = np.zeros((3, 4))       # 二维（3行4列）
zeros_3d = np.zeros((2, 3, 4))    # 三维
print(zeros_2d)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# 全1数组
ones = np.ones((2, 3))
print(ones)
# [[1. 1. 1.]
#  [1. 1. 1.]]

# 填充指定值
full = np.full((2, 3), 7)         # 全部填充为7
print(full)
# [[7 7 7]
#  [7 7 7]]

# 单位矩阵（方阵）
identity = np.eye(4)              # 4×4单位矩阵
print(identity)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]

# 对角矩阵
diag = np.diag([1, 2, 3, 4])
print(diag)
# [[1 0 0 0]
#  [0 2 0 0]
#  [0 0 3 0]
#  [0 0 0 4]]
```

### 2.2.3 范围数组

```python
# arange：类似Python的range
a = np.arange(10)         # 0到9，步长1
print(a)                  # [0 1 2 3 4 5 6 7 8 9]

b = np.arange(1, 10)      # 1到9
print(b)                  # [1 2 3 4 5 6 7 8 9]

c = np.arange(0, 10, 2)   # 0到9，步长2
print(c)                  # [0 2 4 6 8]

# linspace：等差数列（包含端点）
d = np.linspace(0, 1, 5)  # 0到1之间均匀取5个点
print(d)                  # [0.   0.25 0.5  0.75 1.  ]

# logspace：对数等比数列
e = np.logspace(0, 2, 5)  # 10^0到10^2之间取5个点
print(e)                  # [  1.           3.16227766  10.
```

### 2.2.4 随机数组

```python
# 设置随机种子（结果可复现）
np.random.seed(42)

# rand：均匀分布[0, 1)
uniform = np.random.rand(3, 4)
print(uniform)
# [[0.37454012 0.95071431 0.73199394 0.59865848]
#  [0.15601864 0.15599452 0.05808361 0.86617615]
#  [0.60111501 0.70807258 0.02058449 0.96990985]]

# randn：标准正态分布（均值0，方差1）
normal = np.random.randn(3, 4)
print(normal)

# randint：整数随机数组
integers = np.random.randint(0, 10, (3, 4))  # 0到9，随机整数
print(integers)

# choice：随机选择
choices = np.random.choice([1, 2, 3, 4, 5], size=10)
print(choices)

# shuffle：打乱数组
arr = np.arange(10)
np.random.shuffle(arr)
print(arr)
```

## 2.3 数组属性

NumPy数组具有多个重要属性，用于描述数组的结构和特征。

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# shape：形状（元组）
print(arr.shape)    # (2, 3) - 2行3列

# ndim：维度数
print(arr.ndim)     # 2

# size：元素总数
print(arr.size)     # 6

# dtype：数据类型
print(arr.dtype)   # int64

# itemsize：每个元素的字节大小
print(arr.itemsize)  # 8（int64为8字节）

# nbytes：总字节数
print(arr.nbytes)    # 48（6 × 8）

# T：转置
print(arr.T)
# [[1 4]
#  [2 5]
#  [3 6]]
```

**常用数据类型**：

| dtype | 说明 | 字节 |
|-------|------|------|
| int32 | 32位整数 | 4 |
| int64 | 64位整数 | 8 |
| float32 | 32位浮点 | 4 |
| float64 | 64位浮点 | 8 |
| bool | 布尔 | 1 |
| complex64 | 复数（双32位） | 8 |
| complex128 | 复数（双64位） | 16 |

```python
# 指定数据类型创建数组
arr = np.array([1, 2, 3], dtype=np.float32)
print(arr.dtype)  # float32

# 类型转换
arr_float = arr.astype(np.float64)
print(arr_float.dtype)  # float64

arr_int = arr.astype(np.int32)
print(arr_int.dtype)   # int32
```

## 2.4 索引与切片

NumPy数组支持丰富的数据访问方式，包括基本索引、切片、布尔索引和花式索引。

### 2.4.1 基本索引

```python
# 一维数组索引
a = np.array([1, 2, 3, 4, 5])
print(a[0])   # 1（第一个元素）
print(a[-1])  # 5（最后一个元素）

# 二维数组索引
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(b[0])      # [1 2 3]（第一行）
print(b[1, 2])   # 6（第二行第三列）
print(b[-1, -1]) # 9（最后一行最后一列）
```

### 2.4.2 切片操作

切片语法：`start:stop:step`，其中 start 默认为0，stop 默认为该维度大小，step 默认为1。

```python
# 一维数组切片
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print(a[2:7])     # [2 3 4 5 6]（索引2到6）
print(a[:5])      # [0 1 2 3 4]（从头到索引4）
print(a[5:])      # [5 6 7 8 9]（索引5到末尾）
print(a[::2])     # [0 2 4 6 8]（步长2）
print(a[::-1])    # [9 8 7 6 5 4 3 2 1 0]（倒序）

# 二维数组切片
b = np.array([[0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 11]])

print(b[0, :])     # [0 1 2 3]（第一行）
print(b[:, 1])     # [1 5 9]（第二列）
print(b[:2, :2])   # [[0 1] [4 5]]（左上2×2子矩阵）
print(b[1:, 2:])   # [[6 7] [10 11]]（右下2×2子矩阵）
print(b[::2, ::2]) # [[0 2] [8 10]]（跳行跳列）
```

**切片是视图，不是副本**：

```python
# 重要：切片返回的是视图，修改视图会影响原数组
a = np.array([1, 2, 3, 4, 5])
b = a[1:4]
print(b)      # [2 3 4]
b[0] = 99     # 修改视图
print(a)      # [1 99 3 4 5]（原数组被修改！）

# 如果需要副本，使用copy()
c = a[1:4].copy()
c[0] = 100
print(a)      # [1 99 3 4 5]（原数组不变）
```

### 2.4.3 布尔索引

使用布尔数组进行索引，常用于条件筛选。

```python
# 一维布尔索引
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 找出所有偶数
mask = (a % 2 == 0)
print(mask)           # [False  True False  True False  True False  True False  True]
print(a[mask])        # [ 2  4  6  8 10]

# 直接使用条件表达式
print(a[a > 5])       # [ 6  7  8  9 10]
print(a[(a > 2) & (a < 8)])  # [3 4 5 6 7]（&表示AND，|表示OR）

# 二维布尔索引
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 找出大于5的元素
mask = b > 5
print(mask)
# [[False False False]
#  [False False  True]
#  [ True  True  True]]
print(b[mask])  # [6 7 8 9]（返回一维数组）
```

### 2.4.4 花式索引

使用整数数组进行索引，可以不按顺序甚至使用重复索引。

```python
# 一维花式索引
a = np.array([10, 20, 30, 40, 50])

# 按特定顺序获取元素
indices = [0, 2, 4]
print(a[indices])   # [10 30 50]

# 按逆序获取元素
print(a[::-1])      # [50 40 30 20 10]

# 二维花式索引
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 按行索引数组（获取第0行和第2行）
print(b[[0, 2]])    # [[1 2 3] [7 8 9]]

# 使用整数数组指定行和列
rows = np.array([0, 1, 2])
cols = np.array([2, 1, 0])
print(b[rows, cols])  # [3 5 7]（对应位置的元素）

# 使用np.ix_创建网格索引
print(b[np.ix_([0, 2], [0, 2])])
# [[1 3]
#  [7 9]]
```

## 2.5 形状操作

NumPy提供了丰富的形状操作函数，用于改变数组的维度、大小和布局。

### 2.5.1 reshape - 改变形状

```python
a = np.arange(12)
print(a)       # [0 1 2 3 4 5 6 7 8 9 10 11]

# 重塑为3行4列
b = a.reshape(3, 4)
print(b)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# -1自动计算维度
c = a.reshape(3, -1)  # 自动计算列数
print(c.shape)  # (3, 4)

d = a.reshape(-1, 6)  # 自动计算行数
print(d.shape)  # (2, 6)

# 展平为一维
flat = b.reshape(-1)  # 或 b.flatten()
print(flat)  # [0 1 2 3 4 5 6 7 8 9 10 11]
```

**reshape返回视图**：

```python
# reshape通常返回视图
a = np.arange(12)
b = a.reshape(3, 4)
print(b.base is a)  # True（共享数据）

# 验证：修改b会影响a
b[0, 0] = 99
print(a[0])  # 99
```

### 2.5.2 transpose - 转置

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.shape)  # (2, 3)

# 转置
b = a.T
print(b.shape)  # (3, 2)
print(b)
# [[1 4]
#  [2 5]
#  [3 6]]

# 高维数组转置：指定轴的顺序
c = np.arange(24).reshape(2, 3, 4)
print(c.shape)  # (2, 3, 4)

# axes参数指定新轴的顺序
# 原始轴顺序 (0, 1, 2) -> 新顺序 (1, 0, 2)
d = c.transpose(1, 0, 2)
print(d.shape)  # (3, 2, 4)

# np.moveaxis 移动轴
e = np.moveaxis(c, 2, 0)  # 将第2轴移到第0位
print(e.shape)  # (4, 2, 3)
```

### 2.5.3 squeeze与expand_dims

```python
# squeeze：移除大小为1的维度
a = np.arange(12).reshape(1, 3, 4, 1)
print(a.shape)  # (1, 3, 4, 1)

b = np.squeeze(a)
print(b.shape)  # (3, 4)

# 指定移除特定维度
c = np.squeeze(a, axis=0)  # 移除第0维
print(c.shape)  # (3, 4, 1)

# expand_dims：添加大小为1的维度
d = np.array([1, 2, 3])
print(d.shape)  # (3,)

e = np.expand_dims(d, axis=0)  # 添加第0维
print(e.shape)  # (1, 3)

f = np.expand_dims(d, axis=1)  # 添加第1维
print(f.shape)  # (3, 1)
```

### 2.5.4 连接与分割

```python
# 连接
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# concatenate：沿现有轴连接
c = np.concatenate([a, b])
print(c)  # [1 2 3 4 5 6]

# 沿不同轴连接
d = np.array([[1, 2], [3, 4]])
e = np.array([[5, 6], [7, 8]])

f = np.concatenate([d, e], axis=0)  # 垂直连接
print(f)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

g = np.concatenate([d, e], axis=1)  # 水平连接
print(g)
# [[1 2 5 6]
#  [3 4 7 8]]

# vstack：垂直堆叠
h = np.vstack([d, e])  # 等价于concatenate([d, e], axis=0)

# hstack：水平堆叠
i = np.hstack([d, e])  # 等价于concatenate([d, e], axis=1)

# 分割
arr = np.arange(12).reshape(3, 4)
print(arr)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 水平分割为4个数组
split_h = np.hsplit(arr, 4)

# 垂直分割为3个数组
split_v = np.vsplit(arr, 3)

# 指定分割位置
a, b, c = np.split(arr, [1, 3], axis=1)  # 按列索引分割
print(a)  # 第0列
print(b)  # 第1-2列
print(c)  # 第3列
```

## 2.6 数组运算

NumPy的核心优势之一是其向量化运算能力，无需显式循环即可对整个数组进行数学运算。

### 2.6.1 逐元素运算

```python
a = np.array([1, 2, 3, 4, 5])

# 基本算术运算（逐元素应用）
print(a + 10)     # [11 12 13 14 15]
print(a - 5)      # [-4 -3 -2 -1  0]
print(a * 2)      # [ 2  4  6  8 10]
print(a / 2)      # [0.5 1.  1.5 2.  2.5]
print(a ** 2)     # [ 1  4  9 16 25]
print(a % 2)      # [1 0 1 0 1]（取模）

# 两个数组逐元素运算
b = np.array([10, 20, 30, 40, 50])
print(a + b)      # [11 22 33 44 55]
print(a * b)      # [10 40 90 160 250]

# 比较运算
print(a > 3)      # [False False False  True  True]
print(a == 3)     # [False False  True False False]
print(a != 3)     # [True True False True True]
```

**通用函数（ufunc）**：

NumPy的通用函数是一类对数组进行逐元素运算的函数，性能远优于Python循环。

```python
a = np.array([1, 2, 3, 4, 5])

# 三角函数
print(np.sin(a))   # [0.84147098 0.90929743 0.14112001 -0.7568025 -0.95892427]
print(np.cos(a))   # [ 0.54030231 -0.41614684 -0.9899925  -0.65364362  0.28366219]
print(np.tan(a))   # [ 1.55740772 -2.18503986 -0.14254654  1.15782128 -3.38051501]

# 指数与对数
print(np.exp(a))   # [  2.71828183   7.3890561   20.08553692  54.59815003 148.4131591]
print(np.log(a))   # [0.         0.69314718 1.09861229 1.38629436 1.60943791]
print(np.log2(a))  # [0.        1.        1.5849625 2.        2.32192809]
print(np.log10(a)) # [0.         0.30103    0.47712125 0.60205999 0.69897   ]

# 幂函数
print(np.sqrt(a))  # [1.         1.41421356 1.73205081 2.         2.23606798]

# 绝对值
c = np.array([-1, -2, -3, 4, 5])
print(np.abs(c))   # [1 2 3 4 5]

# 四舍五入
d = np.array([1.234, 2.567, 3.891])
print(np.round(d, 1))  # [1.2 2.6 3.9]
print(np.floor(d))     # [1. 2. 3.]（向下取整）
print(np.ceil(d))      # [2. 3. 4.]（向上取整）
```

### 2.6.2 矩阵运算

NumPy使用 `@` 运算符或 `np.dot()` 函数进行矩阵乘法。

```python
# 二维矩阵乘法
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# 矩阵乘法（维度必须兼容）
C = A @ B
print(C)
# [[1*5+2*7 1*6+2*8]   [[19 22]
#  [3*5+4*7 3*6+4*8]] = [43 50]]

# 或使用np.dot
D = np.dot(A, B)
print(np.array_equal(C, D))  # True

# 向量点积
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot = np.dot(v1, v2)
print(dot)  # 1*4 + 2*5 + 3*6 = 32

# 向量范数
norm = np.linalg.norm(v1)  # 默认L2范数
print(norm)  # sqrt(1+4+9) = 3.7416...

# 矩阵的转置
print(A.T)
# [[1 3]
#  [2 4]]
```

### 2.6.3 广播机制

广播（Broadcasting）是NumPy处理不同形状数组运算的机制，使运算更加便捷。

**广播规则**：

1. 让所有输入数组向形状相同的方向扩展
2. 每个维度的大小要么相同，要么其中一个为1
3. 从右向左比较维度

```python
# 基本广播示例
a = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape (2, 3)

b = np.array([10, 20, 30])  # shape (3,)

# b被广播为 [[10, 20, 30], [10, 20, 30]]
result = a + b
print(result)
# [[11 22 33]
#  [14 25 36]]

# 行向量广播
c = np.array([[1], [2]])  # shape (2, 1)
# c被广播为 [[1, 1, 1], [2, 2, 2]]
result = a + c
print(result)
# [[ 2  3  4]
#  [ 6  7  8]]

# 标量与数组运算（自动广播）
d = 100
result = a + d  # d被广播为与a相同形状的数组
print(result)
# [[101 102 103]
#  [104 105 106]]
```

**更复杂的广播示例**：

```python
# 图像数据处理示例
# 假设有batch_size=3, height=4, width=5, channels=3 的图像
images = np.random.randn(3, 4, 5, 3)  # (3, 4, 5, 3)

# 减去均值（沿最后一维计算）
mean = np.mean(images, axis=-1, keepdims=True)  # (3, 4, 5, 1)
normalized = images - mean
print(normalized.shape)  # (3, 4, 5, 3)

# 加权求和示例
weights = np.array([0.2, 0.5, 0.3])  # (3,)
# weights被广播为 (3, 4, 5, 3) 进行逐元素乘法
weighted = images * weights.reshape(1, 1, 1, 3)
print(weighted.shape)  # (3, 4, 5, 3)
```

### 2.6.4 归约操作

归约操作将数组缩减为更小的维度或单一值。

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# 求和
print(np.sum(a))         # 21（所有元素之和）
print(np.sum(a, axis=0)) # [5 7 9]（按列求和）
print(np.sum(a, axis=1)) # [ 6 15]（按行求和）

# 均值
print(np.mean(a))         # 3.5
print(np.mean(a, axis=0)) # [2.5 3.5 4.5]

# 标准差
print(np.std(a))          # 约1.707

# 最大最小值
print(np.min(a))          # 1
print(np.max(a))          # 6
print(np.argmin(a))       # 0（最小值的索引）
print(np.argmax(a))       # 5（最大值的索引）

# 累积操作
print(np.cumsum(a))       # [ 1  3  6 10 15 21]
print(np.cumprod(a))      # [  1   2   6  24 120 720]
```

### 2.6.5 比较与掩码操作

```python
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 找出满足条件的元素索引
print(np.where(a > 5))    # (array([5, 6, 7, 8, 9]),)

# 条件替换
b = np.where(a > 5, a, 0)  # 大于5保留，否则为0
print(b)                   # [0 0 0 0 0 6 7 8 9 10]

# 判断是否全部/任意满足条件
c = np.array([[1, 2, 3],
              [4, 5, 6]])
print(np.all(c > 0))  # True（所有元素都大于0）
print(np.any(c > 5))  # True（存在大于5的元素）

# 排序
d = np.array([3, 1, 4, 1, 5, 9, 2, 6])
sorted_idx = np.argsort(d)  # 返回排序后的索引
print(d[sorted_idx])  # [1 1 2 3 4 5 6 9]
```

## 2.7 实用技巧

### 2.7.1 常用函数速查

```python
# 数组复制
a = np.array([1, 2, 3])
b = a.copy()  # 完全独立的副本

# 排序
a = np.array([3, 1, 2])
sorted_a = np.sort(a)  # 返回副本
a.sort()  # 原地排序

# 唯一值
a = np.array([1, 2, 2, 3, 3, 3])
unique = np.unique(a)  # [1 2 3]

# 集合运算
a = np.array([1, 2, 3, 4])
b = np.array([3, 4, 5, 6])
print(np.intersect1d(a, b))  # [3 4]（交集）
print(np.union1d(a, b))     # [1 2 3 4 5 6]（并集）
print(np.setdiff1d(a, b))   # [1 2]（差集）

# 重排数组
a = np.arange(10)
np.random.shuffle(a)  # 原地打乱
print(a)
```

### 2.7.2 文件I/O

```python
# 保存数组到文件
a = np.arange(12).reshape(3, 4)

# 保存为文本格式
np.savetxt("array.txt", a)

# 保存为二进制格式（.npy）
np.save("array.npy", a)

# 保存多个数组
np.savez("arrays.npz", a=a, data=a * 2)

# 加载数组
loaded = np.load("array.npy")
print(np.array_equal(a, loaded))  # True

# 加载多个数组
multi = np.load("arrays.npz")
print(multi["a"])      # 原始数组
print(multi["data"])   # data数组
```

### 2.7.3 性能注意事项

```python
# 避免循环，使用向量化
# 慢：
result = []
for i in range(1000):
    result.append(i ** 2)
result = np.array(result)

# 快：
result = np.arange(1000) ** 2

# 使用原地操作节省内存
a = np.arange(1000000)
# 慢：创建新数组
b = a + 1
# 快：原地操作
np.add(a, 1, out=a)

# 使用np.einsum进行高效矩阵运算
A = np.random.randn(100, 200)
B = np.random.randn(200, 50)

# 等价于 C = A @ B，但更灵活
C = np.einsum('ij,jk->ik', A, B)

# 使用np.take代替花式索引（某些情况下更快）
indices = np.array([0, 5, 10])
a = np.arange(10000)
# 慢：a[indices]
# 快：np.take(a, indices)
```

### 2.7.4 常见错误处理

```python
# 形状不匹配
a = np.array([1, 2, 3])
b = np.array([4, 5])
try:
    c = a + b  # ValueError
except ValueError as e:
    print(f"形状不匹配: {e}")

# 正确做法：使用广播或重塑
b_reshaped = b.reshape(1, 2)  # 或 b.reshape(-1, 1)
c = a.reshape(3, 1) + b
print(c)
# [[5 6]
#  [6 7]
#  [7 8]]

# 类型错误
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([1.5, 2.5, 3.5], dtype=np.float64)
c = a + b  # int32转换为float64后运算
print(c)  # [2.5 4.5 6.5]
```

## 本章小结

本章介绍了NumPy的基础知识，主要包括：

1. **NumPy数组创建**：从Python列表转换、预定义数组（zeros、ones、eye等）、范围数组（arange、linspace）、随机数组。

2. **数组属性**：shape、ndim、size、dtype、itemsize等核心属性，以及数据类型转换。

3. **索引与切片**：基本索引、多维数组索引、切片（返回视图）、布尔索引（条件筛选）、花式索引（整数数组索引）。

4. **形状操作**：reshape改变形状、transpose转置、squeeze移除单维度、expand_dims添加单维度、连接与分割操作。

5. **数组运算**：逐元素运算（算术、比较、通用函数）、矩阵乘法、广播机制（核心概念）、归约操作（sum、mean、max等）、比较与掩码。

6. **实用技巧**：常用函数速查、文件I/O（保存和加载）、性能优化建议、常见错误处理。

NumPy的高性能数组操作是后续学习PyTorch的基础，很多概念和API设计在PyTorch中都有相似体现。

## 思考与练习

1. 创建一个 5×5 的矩阵，元素为 1 到 25 的整数，然后：
   - 提取主对角线元素
   - 提取上三角部分
   - 计算矩阵的转置

2. 给定两个数组 `a = np.array([1, 2, 3, 4, 5])` 和 `b = np.array([5, 4, 3, 2, 1])`，使用NumPy计算它们的相关系数（Pearson相关系数），不使用Python循环。

3. 有一个形状为 (100, 32, 32, 3) 的图像数据集（100张32×32的RGB图像），请：
   - 计算每张图像的均值和标准差
   - 对所有图像按通道计算均值（形状为(3,)的数组）
   - 将像素值归一化到 [0, 1] 范围

4. 实现一个函数，输入一个二维NumPy数组，返回其主对角线元素之和（矩阵迹）。分别用 `np.trace` 和手动实现两种方式完成。

5. 给定一个随机数组，使用NumPy找出其中第k大的元素，要求时间复杂度优于O(n log n)排序。
