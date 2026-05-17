# Python基础语法

## 学习目标

- 理解Python中的变量与基本数据类型
- 掌握条件判断和循环控制流的写法
- 能够定义和调用函数
- 了解面向对象编程的核心概念：类与对象
- 掌握继承与多态的实现方式
- 理解列表推导式、生成器、迭代器和装饰器等高级特性

## 1.1 变量与数据类型

### 1.1.1 变量概述

变量（Variable）是编程中最基础的概念之一，用于存储数据。在Python中，变量不需要预先声明类型，解释器会根据赋值自动推断其类型。这种特性称为**动态类型（Dynamic Typing）**。

```python
# 变量赋值示例
name = "Alice"      # 字符串类型
age = 25            # 整数类型
height = 1.68       # 浮点数类型
is_student = True   # 布尔类型
```

上述代码演示了Python中最常用的几种基本数据类型。变量名 `name`、`age`、`height`、`is_student` 就像一个个容器，每个容器存放着不同类型的数据。

### 1.1.2 基本数据类型

Python内置了丰富的数据类型，主要分为以下几类：

#### 数值类型

**整数（int）**：表示没有小数部分的数值，范围理论上无限制。

```python
a = 100
b = -50
c = 0

# 不同进制的表示
binary_num = 0b1010      # 二进制：10
octal_num = 0o755       # 八进制：493
hex_num = 0xFF          # 十六进制：255

# 大整数运算
large_num = 10 ** 100   # 10的100次方
print(large_num)        # 输出：100000000...
```

**浮点数（float）**：表示带小数部分的数值，采用IEEE 754双精度标准。

```python
pi = 3.14159
gravity = 9.8
negative = -0.5

# 科学计数法
speed_of_light = 3e8    # 3 × 10^8
```

**复数（complex）**：Python原生支持复数运算。

```python
z = 3 + 4j
print(z.real)  # 输出：3.0
print(z.imag)  # 输出：4.0
```

#### 字符串（str）

字符串是字符的序列，用单引号或双引号包围。

```python
# 字符串定义
single_quote = 'Hello'
double_quote = "World"
multi_line = """这是
一个多行
字符串"""

# 字符串拼接
greeting = "Hello" + ", " + "World"
print(greeting)  # 输出：Hello, World

# 字符串重复
echo = "Ha" * 3
print(echo)  # 输出：HaHaHa

# 字符串长度
text = "Python"
print(len(text))  # 输出：6

# 字符串索引与切片
first_char = text[0]    # 'P'
substring = text[0:3]   # 'Pyt'
reversed_text = text[::-1]  # 'nohtyP'
```

**转义字符**：用反斜杠 `\` 表示特殊含义的字符。

```python
new_line = "第一行\n第二行"    # 换行
tab_indent = "列1\t列2"       # 制表符
quote_in_string = "她说\"你好\""  # 引号
backslash_path = "C\\Users\\name"  # 反斜杠
```

**字符串方法**：Python为字符串提供了丰富的内置方法。

```python
s = "  Hello, Python!  "

# 大小写转换
print(s.upper())      # '  HELLO, PYTHON!  '
print(s.lower())      # '  hello, python!  '
print(s.capitalize())  # '  hello, python!  '

# 去除空白
print(s.strip())      # 'Hello, Python!'
print(s.lstrip())     # 'Hello, Python!  '
print(s.rstrip())     # '  Hello, Python!'

# 查找与替换
print(s.find("Python"))   # 9（首次出现的索引）
print(s.replace("Python", "World"))  # '  Hello, World!  '

# 分割与连接
words = "apple,banana,orange".split(",")  # ['apple', 'banana', 'orange']
joined = "-".join(words)                   # 'apple-banana-orange'

# 格式化
name = "Alice"
age = 25
formatted = f"姓名：{name}，年龄：{age}"  # f-string格式化
print(formatted)  # 输出：姓名：Alice，年龄：25
```

#### 布尔类型（bool）

布尔类型只有两个值：`True`（真）和 `False`（假）。

```python
is_valid = True
is_empty = False

# 布尔运算
print(True and False)   # False
print(True or False)    # True
print(not True)         # False
```

在Python中，以下值在布尔上下文中被视为**假（False）**：
- `None`
- `False`
- 数值零：`0`、`0.0`、`0j`
- 空序列：`''`、`[]`、`()`
- 空字典：`{}`
- 自定义对象，其 `__bool__()` 返回 `False` 或 `__len__()` 返回 `0`

#### None类型

`None` 表示空值或 absence of value，类似于其他语言中的 `null`。

```python
result = None
if result is None:
    print("没有结果")
```

**注意**：比较 `None` 时应使用 `is None` 或 `is not None`，而不是 `== None` 或 `!= None`，因为 `is` 比较的是对象身份，而 `==` 比较的是值相等性。

### 1.1.3 类型转换

Python提供了显式转换数据类型的能力。

```python
# 字符串转整数
num_str = "42"
num = int(num_str)      # 42
num_float = float(num_str)  # 42.0

# 整数转字符串
age = 25
age_str = str(age)     # "25"

# 字符串转浮点数
pi_str = "3.14"
pi = float(pi_str)     # 3.14

# 浮点数转整数（截断小数部分）
x = 3.7
print(int(x))          # 3（不是四舍五入！）

# 四舍五入
print(round(3.7))      # 4
print(round(3.14159, 2))  # 3.14（保留两位小数）

# 转换为布尔
print(bool(0))         # False
print(bool(1))         # True
print(bool(""))        # False
print(bool("hello"))   # True
```

### 1.1.4 常量

Python中没有真正的常量机制，但通常使用全大写变量名表示"不应被修改"的量。

```python
# 约定：全大写表示常量
MAX_RETRY_COUNT = 3
PI = 3.14159
GRAVITY = 9.8
```

## 1.2 控制流

控制流（Control Flow）决定了程序执行的顺序。Python使用缩进（通常为4个空格）来表示代码块的层级关系。

### 1.2.1 条件判断（if语句）

```python
# 基础if语句
age = 18

if age >= 18:
    print("已成年")
```

```python
# if-else语句
temperature = 30

if temperature > 35:
    print("高温预警")
else:
    print("温度正常")
```

```python
# if-elif-else语句（多条件）
score = 85

if score >= 90:
    grade = "A"
    print("优秀")
elif score >= 80:
    grade = "B"
    print("良好")
elif score >= 70:
    grade = "C"
    print("中等")
elif score >= 60:
    grade = "D"
    print("及格")
else:
    grade = "F"
    print("不及格")

print(f"等级：{grade}")
```

**三元表达式**：Python支持单行条件表达式。

```python
age = 20
status = "成年" if age >= 18 else "未成年"
print(status)  # 输出：成年
```

### 1.2.2 循环（for循环）

`for` 循环用于遍历可迭代对象（Iterable），如列表、字符串、字典等。

```python
# 遍历列表
fruits = ["苹果", "香蕉", "橙子"]

for fruit in fruits:
    print(fruit)

# 遍历字符串
for char in "Python":
    print(char)

# 使用range()函数
for i in range(5):
    print(i)          # 输出：0, 1, 2, 3, 4

for i in range(1, 6):
    print(i)          # 输出：1, 2, 3, 4, 5

for i in range(0, 10, 2):
    print(i)          # 输出：0, 2, 4, 6, 8（偶数）

# 遍历字典
person = {"name": "Alice", "age": 25, "city": "北京"}

for key in person:
    print(f"{key}: {person[key]}")

for key, value in person.items():
    print(f"{key}: {value}")

for value in person.values():
    print(value)
```

**enumerate()函数**：同时获取索引和值。

```python
fruits = ["苹果", "香蕉", "橙子"]

for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
# 输出：
# 0: 苹果
# 1: 香蕉
# 2: 橙子
```

### 1.2.3 循环（while循环）

`while` 循环在条件为真时重复执行代码块。

```python
# 基础while循环
count = 0
while count < 5:
    print(count)
    count += 1

# while循环与break
while True:
    user_input = input("输入 'quit' 退出：")
    if user_input == "quit":
        break
    print(f"你输入了：{user_input}")

# while循环与continue
i = 0
while i < 10:
    i += 1
    if i % 2 == 0:
        continue  # 跳过偶数
    print(i)  # 输出：1, 3, 5, 7, 9
```

### 1.2.4 循环的else子句

Python特有的语法：循环正常结束时会执行 `else` 子句（如果循环被 `break` 终止则不执行）。

```python
# 搜索示例
numbers = [1, 3, 5, 7, 9]
target = 6

for n in numbers:
    if n == target:
        print(f"找到{target}")
        break
else:
    print(f"未找到{target}")
```

### 1.2.5 嵌套循环

循环内部可以包含另一个循环。

```python
# 打印九九乘法表
for i in range(1, 10):
    for j in range(1, i + 1):
        print(f"{j}×{i}={i*j}", end="\t")
    print()  # 换行
```

## 1.3 函数定义

函数（Function）是一段可重复使用的代码，用于完成特定任务。

### 1.3.1 函数基础

```python
# 定义函数
def greet():
    """问候函数"""
    print("你好！")

# 调用函数
greet()  # 输出：你好！

# 带参数的函数
def greet_person(name):
    """向指定人打招呼"""
    print(f"你好，{name}！")

greet_person("Alice")  # 输出：你好，Alice！

# 带返回值的函数
def add(a, b):
    """返回两数之和"""
    return a + b

result = add(3, 5)
print(result)  # 8

# 多返回值函数
def divide(a, b):
    """返回商和余数"""
    quotient = a // b
    remainder = a % b
    return quotient, remainder

q, r = divide(10, 3)
print(f"商：{q}，余数：{r}")  # 商：3，余数：1
```

### 1.3.2 参数默认值

```python
def power(base, exponent=2):
    """计算幂次，默认平方"""
    return base ** exponent

print(power(3))     # 9（使用默认值）
print(power(3, 3))  # 27（使用指定值）
```

**注意**：带默认值的参数必须放在无默认值参数的后面。

```python
# 错误示例
# def func(a=1, b):  # SyntaxError
#     pass

# 正确示例
def func(a, b=2):
    pass
```

### 1.3.3 关键字参数

使用参数名指定参数值，可以不按顺序传递。

```python
def describe_pet(animal, name):
    print(f"我有一只{animal}，名叫{name}")

describe_pet(animal="猫", name="小白")  # 我有一只猫，名叫小白
describe_pet(name="小黑", animal="狗")  # 我有一只狗，名叫小黑
```

### 1.3.4 可变参数

使用 `*args` 和 `**kwargs` 处理不确定数量的参数。

```python
# *args：接收任意数量的位置参数（元组）
def sum_all(*numbers):
    """计算所有参数的和"""
    total = 0
    for n in numbers:
        total += n
    return total

print(sum_all(1, 2, 3))        # 6
print(sum_all(1, 2, 3, 4, 5))  # 15

# **kwargs：接收任意数量的关键字参数（字典）
def print_info(**info):
    """打印所有关键字参数"""
    for key, value in info.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="北京")
# 输出：
# name: Alice
# age: 25
# city: 北京
```

### 1.3.5 参数类型注解

Python 3.5+ 支持类型注解（Type Hints），使代码更易读。

```python
def greet(name: str) -> str:
    """返回问候语"""
    return f"你好，{name}！"

def add(a: int, b: int) -> int:
    return a + b

# 列表类型注解
from typing import List

def process_numbers(numbers: List[int]) -> List[int]:
    return [n * 2 for n in numbers]
```

### 1.3.6 变量作用域

Python中的作用域遵循 LEGB 规则：Local（局部）→ Enclosing（闭包）→ Global（全局）→ Built-in（内置）。

```python
# 全局变量
count = 10

def modify_global():
    global count  # 声明使用全局变量
    count = 20

def read_only():
    print(count)  # 读取全局变量

modify_global()
read_only()  # 输出：20
```

**闭包**：内层函数可以访问外层函数的变量。

```python
def outer(x):
    def inner(y):
        return x + y
    return inner

add_5 = outer(5)
print(add_5(3))   # 8
print(add_5(10))  # 15
```

### 1.3.7 匿名函数（lambda）

`lambda` 用于创建简短的匿名函数。

```python
# 基础lambda
square = lambda x: x ** 2
print(square(5))  # 25

# 多参数lambda
add = lambda a, b: a + b
print(add(3, 4))  # 7

# 与内置函数配合使用
numbers = [1, 2, 3, 4, 5]

# 按绝对值排序
sorted_nums = sorted(numbers, key=lambda x: -x)
print(sorted_nums)  # [5, 4, 3, 2, 1]

# map：应用函数到每个元素
doubled = list(map(lambda x: x * 2, numbers))
print(doubled)  # [2, 4, 6, 8, 10]

# filter：过滤元素
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4]
```

## 1.4 面向对象基础

面向对象编程（Object-Oriented Programming，OOP）是一种组织代码的方式，通过"类"定义对象的结构和行为。

### 1.4.1 类与对象

**类（Class）**：对象的蓝图或模板，定义了对象的属性和方法。

**对象（Object）**：类的实例，具有具体的状态和行为。

```python
# 定义类
class Dog:
    """狗类"""

    # 类属性（所有实例共享）
    species = "犬科"

    # 初始化方法（构造函数）
    def __init__(self, name, age):
        # 实例属性
        self.name = name
        self.age = age

    # 实例方法
    def bark(self):
        """狗叫"""
        return f"{self.name}汪汪叫！"

    def get_info(self):
        """获取信息"""
        return f"{self.name}，{self.age}岁"

# 创建对象（实例化）
dog1 = Dog("小白", 3)
dog2 = Dog("小黑", 5)

# 访问属性
print(dog1.name)     # 小白
print(dog2.age)     # 5

# 调用方法
print(dog1.bark())   # 小白汪汪叫！

# 类属性
print(Dog.species)   # 犬科
```

### 1.4.2 面向对象的三大特性

#### 封装（Encapsulation）

封装将数据和操作封装在类内部，隐藏内部实现细节。

```python
class BankAccount:
    """银行账户类"""

    def __init__(self, owner, balance=0):
        self.owner = owner
        # 私有属性（以双下划线开头）
        self.__balance = balance

    # 公开方法访问私有属性
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return True
        return False

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return True
        return False

    def get_balance(self):
        return self.__balance

# 使用
account = BankAccount("Alice", 1000)
print(account.get_balance())  # 1000
account.deposit(500)
print(account.get_balance())  # 1500
account.withdraw(200)
print(account.get_balance())  # 1300

# 无法直接访问私有属性
# print(account.__balance)  # AttributeError
```

#### 继承（Inheritance）

继承允许创建基于现有类的新类，子类自动获得父类的属性和方法。

```python
# 父类
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("子类必须实现speak方法")

    def info(self):
        return f"名字：{self.name}"

# 子类
class Cat(Animal):
    def speak(self):
        return f"{self.name}喵喵叫"

class Dog(Animal):
    def speak(self):
        return f"{self.name}汪汪叫"

class Bird(Animal):
    def speak(self):
        return f"{self.name}叽叽叫"

# 创建子类实例
cat = Cat("小白")
dog = Dog("小黑")
bird = Bird("小小")

print(cat.speak())    # 小白喵喵叫
print(dog.speak())    # 小黑汪汪叫
print(bird.speak())   # 小小叽叽叫
```

#### 多态（Polymorphism）

多态允许不同类的对象对同一消息（方法调用）做出不同响应。

```python
# 统一接口处理不同对象
def make_speak(animal):
    print(animal.speak())

animals = [Cat("白猫"), Dog("黑狗"), Bird("小鸟")]

for animal in animals:
    make_speak(animal)
# 输出：
# 白猫喵喵叫
# 黑狗汪汪叫
# 小鸟叽叽叫
```

### 1.4.3 方法的类型

```python
class MyClass:
    class_attr = "类属性"  # 类属性

    def __init__(self, value):
        self.instance_attr = value  # 实例属性

    # 实例方法：第一个参数是self
    def instance_method(self):
        return f"实例方法，访问：{self.instance_attr}"

    # 类方法：第一个参数是cls
    @classmethod
    def class_method(cls):
        return f"类方法，访问：{cls.class_attr}"

    # 静态方法：没有self或cls参数
    @staticmethod
    def static_method():
        return "静态方法，不访问任何实例或类属性"

# 调用
obj = MyClass("value")
print(obj.instance_method())  # 实例方法，访问：value
print(MyClass.class_method())  # 类方法，访问：类属性
print(MyClass.static_method()) # 静态方法，不访问任何实例或类属性
```

### 1.4.4 属性装饰器（@property）

使用 `@property` 将方法转换为只读属性。

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        """圆的半径"""
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("半径不能为负数")
        self._radius = value

    @property
    def area(self):
        """圆的面积（只读）"""
        return 3.14159 * self._radius ** 2

# 使用
circle = Circle(5)
print(circle.radius)  # 5（像访问属性一样）
print(circle.area)    # 78.53975

circle.radius = 10    # 修改半径
print(circle.area)    # 314.159

# circle.radius = -5  # ValueError: 半径不能为负数
```

## 1.5 Python高级特性

### 1.5.1 列表推导式

列表推导式（List Comprehension）是一种简洁的创建列表的方式。

```python
# 基础语法
# [表达式 for 变量 in 可迭代对象]

# 传统方式
squares = []
for i in range(1, 6):
    squares.append(i ** 2)
print(squares)  # [1, 4, 9, 16, 25]

# 列表推导式
squares = [i ** 2 for i in range(1, 6)]
print(squares)  # [1, 4, 9, 16, 25]

# 带条件筛选
evens = [i for i in range(1, 11) if i % 2 == 0]
print(evens)  # [2, 4, 6, 8, 10]

# 嵌套列表推导式
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 带if-else的列表推导式
labels = ["正数" if x > 0 else "非正数" for x in [-3, -1, 0, 2, 5]]
print(labels)  # ['非正数', '非正数', '非正数', '正数', '正数']
```

**其他推导式**：

```python
# 字典推导式
squares_dict = {i: i ** 2 for i in range(1, 6)}
print(squares_dict)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# 集合推导式
unique_squares = {x ** 2 for x in [-2, -1, 0, 1, 2]}
print(unique_squares)  # {0, 1, 4}

# 生成器表达式（惰性求值）
gen = (x ** 2 for x in range(1, 1000001))
print(next(gen))  # 1
print(next(gen))  # 4
# 注意：生成器只能遍历一次
```

### 1.5.2 生成器与迭代器

**迭代器（Iterator）**：实现了 `__iter__()` 和 `__next__()` 方法的对象。

```python
# 使用iter()和next()
my_list = [1, 2, 3]
iterator = iter(my_list)

print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
# print(next(iterator))  # StopIteration
```

**生成器（Generator）**：使用 `yield` 关键字的函数，每次调用返回一个值并暂停执行状态。

```python
# 生成器函数
def count_up_to(n):
    """计数生成器"""
    count = 1
    while count <= n:
        yield count
        count += 1

# 使用生成器
counter = count_up_to(5)
print(next(counter))  # 1
print(next(counter))  # 2
print(next(counter))  # 3

# 遍历生成器
for num in count_up_to(3):
    print(num)
# 输出：1, 2, 3

# 生成器表达式（惰性求值，节省内存）
def fibonacci():
    """斐波那契数列生成器"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()
for _ in range(10):
    print(next(fib), end=" ")
# 输出：0 1 1 2 3 5 8 13 21 34
```

**生成器 vs 列表**：

```python
import sys

# 列表：一次性占用所有内存
my_list = [i for i in range(10000)]
print(sys.getsizeof(my_list), "bytes")  # 约80000 bytes

# 生成器：惰性求值，占用少量内存
my_gen = (i for i in range(10000))
print(sys.getsizeof(my_gen), "bytes")  # 约200 bytes
```

### 1.5.3 装饰器

装饰器（Decorator）是一种修改函数或类行为的高级语法，可以在不修改原函数的情况下添加额外功能。

```python
# 简单装饰器
def my_decorator(func):
    """装饰器函数"""
    def wrapper(*args, **kwargs):
        print("函数执行前")
        result = func(*args, **kwargs)
        print("函数执行后")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("你好！")

say_hello()
# 输出：
# 函数执行前
# 你好！
# 函数执行后
```

**带参数的装饰器**：

```python
import time

def timer(seconds):
    """计时器装饰器工厂"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"函数执行耗时：{elapsed:.2f}秒")
            return result
        return wrapper
    return decorator

@timer(2)
def slow_function():
    time.sleep(1)
    print("函数执行完成")

slow_function()
```

**类装饰器**：

```python
class CallCounter:
    """统计函数调用次数"""
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"函数被调用了{self.count}次")
        return self.func(*args, **kwargs)

@CallCounter
def greet():
    print("你好！")

greet()  # 函数被调用了1次，你好！
greet()  # 函数被调用了2次，你好！
```

**functools.wraps 保留原函数信息**：

```python
import functools

def log_calls(func):
    @functools.wraps(func)  # 保留原函数元信息
    def wrapper(*args, **kwargs):
        print(f"调用 {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_calls
def add(a, b):
    """返回两数之和"""
    return a + b

print(add.__name__)  # add（而不是wrapper）
print(add.__doc__)   # 返回两数之和
```

### 1.5.4 上下文管理器

上下文管理器（Context Manager）用于管理资源，确保资源正确获取和释放。

```python
# 使用with语句
with open("example.txt", "w") as f:
    f.write("Hello, World!")
# 文件自动关闭，无需f.close()

# try-finally等价
f = open("example.txt", "w")
try:
    f.write("Hello, World!")
finally:
    f.close()
```

**自定义上下文管理器**：

```python
# 方法1：使用类
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False  # 不屏蔽异常

with FileManager("test.txt", "w") as f:
    f.write("Hello!")

# 方法2：使用contextlib
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    f = open(filename, mode)
    try:
        yield f
    finally:
        f.close()
```

## 本章小结

本章介绍了Python编程的基础知识，主要包括：

1. **变量与数据类型**：Python的动态类型系统、基本数据类型（整数、浮点数、字符串、布尔值、None）、类型转换规则。

2. **控制流**：条件判断（if-elif-else）、循环（for、while）、break/continue语句、Python特有的循环else子句。

3. **函数定义**：函数的基本结构、参数类型（默认值、关键字参数、可变参数 *args 和 **kwargs）、类型注解、变量作用域、lambda表达式。

4. **面向对象编程**：类和对象的概念、封装（私有属性）、继承（子类化）、多态（统一接口）、实例方法/类方法/静态方法的区别、@property装饰器。

5. **Python高级特性**：列表推导式、生成器与迭代器、装饰器模式、上下文管理器。

这些知识构成了Python编程的核心基础，掌握它们为后续学习NumPy和PyTorch等数据科学工具打下坚实基础。

## 思考与练习

1. 编写一个函数，接受一个整数列表，返回其中所有奇数的和。例如，输入 `[1, 2, 3, 4, 5]`，输出 `9`（1+3+5）。

2. 创建一个 `Rectangle` 类，包含 `width` 和 `height` 属性，以及计算面积和周长的方法。添加属性验证，确保宽高为正数。

3. 使用列表推导式将 `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` 展平为一维列表 `[1, 2, 3, 4, 5, 6, 7, 8, 9]`。

4. 编写一个装饰器 `retry`，在函数执行失败时自动重试最多3次，每次重试间隔1秒。适用于网络请求等可能临时失败的操作。

5. 实现一个生成器函数 `primes(limit)`，返回小于 `limit` 的所有质数。要求使用埃拉托斯特尼筛法（Sieve of Eratosthenes）算法实现。
