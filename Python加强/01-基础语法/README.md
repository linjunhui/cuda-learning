# Python基础语法学习

## 学习目标

熟练掌握Python基本语法和特性，为后续学习打下坚实基础。

## 学习时间

1个月（30天）

## 学习内容

### 第1周：Python核心语法
- Python基础语法（变量、数据类型、运算符）
  - 变量命名规则和命名规范
  - 基本数据类型（int, float, str, bool）
  - 类型转换和类型检查
  - 运算符（算术、比较、逻辑、赋值、位运算）
  - 运算符优先级
- 控制流语句
  - 条件语句（if, elif, else）
  - 循环语句（for, while）
  - 循环控制（break, continue, else）
  - 嵌套控制结构
- 函数定义和调用
  - 函数定义语法
  - 参数传递（位置参数、关键字参数、默认参数）
  - 可变参数（*args, **kwargs）
  - 返回值（return语句）
  - 函数作用域和命名空间
  - Lambda函数
  - 函数文档字符串（docstring）

### 第2周：内置数据结构
- 列表（list）
  - 列表创建和初始化
  - 列表操作（增删改查）
  - 列表推导式
  - 切片操作
  - 列表方法（append, extend, insert, remove, pop等）
- 字典（dict）
  - 字典创建和初始化
  - 字典操作（增删改查）
  - 字典推导式
  - 字典方法（keys, values, items, get, update等）
  - 字典视图对象
- 集合（set）
  - 集合创建和初始化
  - 集合操作（交集、并集、差集）
  - 集合推导式
  - 集合方法（add, remove, discard, union, intersection等）
- 元组（tuple）
  - 元组创建和初始化
  - 元组操作（不可变特性）
  - 命名元组（namedtuple）
  - 元组解包

### 第3周：文件操作和异常处理
- 文件操作
  - 文件打开和关闭（open, close, with语句）
  - 文件读写模式（r, w, a, x, b, t）
  - 文件读取方法（read, readline, readlines）
  - 文件写入方法（write, writelines）
  - 文件指针操作（seek, tell）
  - 路径操作（os.path, pathlib）
- 异常处理
  - 异常概念和类型
  - try-except语句
  - 异常捕获和处理
  - 多个except子句
  - else和finally子句
  - 异常抛出（raise）
  - 自定义异常类
  - 异常链（from关键字）
- 上下文管理器
  - with语句的使用
  - 上下文管理器协议（__enter__, __exit__）

### 第4周：模块和包
- 模块（module）
  - 模块导入（import, from...import）
  - 模块搜索路径（sys.path）
  - 模块重载（reload）
  - 模块属性（__name__, __file__, __doc__）
  - 标准库模块（os, sys, datetime, json等）
- 包（package）
  - 包的创建和结构
  - 包的导入
  - __init__.py文件
  - 相对导入和绝对导入
  - 包的组织方式
- 常用标准库
  - os：操作系统接口
  - sys：系统相关参数和函数
  - datetime：日期和时间处理
  - json：JSON数据处理
  - re：正则表达式
  - collections：特殊容器类型
  - itertools：迭代器工具

## 实践项目

### 项目1：数据处理脚本（第1周）
实现一个简单的数据处理脚本：
- 读取数据文件
- 数据清洗和转换
- 数据统计和汇总
- 结果输出

### 项目2：文件管理工具（第2-3周）
编写文件操作和异常处理程序：
- 文件复制和移动
- 文件内容搜索
- 异常安全编程
- 日志记录

### 项目3：模块化程序（第4周）
编写一个模块化的Python程序：
- 创建自定义包
- 模块化设计
- 标准库使用
- 程序组织

## 学习资源

### 书籍
- 《Python编程：从入门到实践》（第1-11章）
- 《流畅的Python》（第1-3章）

### 在线资源
- Python官方文档
- Real Python教程
- Python Tutorial

### 练习平台
- LeetCode（基础题目）
- HackerRank
- Codewars

## 每日学习计划

### 工作日（2-3小时）
- 理论学习：1小时
- 编程实践：1-2小时

### 周末（4-6小时）
- 理论学习：2小时
- 编程实践：2-4小时

## 检查点

### 第1周检查点
- [ ] 掌握Python基本语法
- [ ] 能够编写简单的Python程序
- [ ] 理解函数和作用域概念
- [ ] 完成项目1

### 第2周检查点
- [ ] 熟练使用Python内置数据结构
- [ ] 掌握列表推导式和字典推导式
- [ ] 能够选择合适的数据结构

### 第3周检查点
- [ ] 掌握文件操作
- [ ] 理解异常处理机制
- [ ] 能够编写异常安全的程序
- [ ] 完成项目2

### 第4周检查点
- [ ] 理解模块和包的概念
- [ ] 能够创建和使用模块
- [ ] 熟悉常用标准库
- [ ] 完成项目3

## 代码示例

### 列表推导式示例
```python
# 列表推导式
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# 字典推导式
square_dict = {x: x**2 for x in range(10)}
```

### 异常处理示例
```python
try:
    with open('file.txt', 'r') as f:
        content = f.read()
except FileNotFoundError:
    print("文件不存在")
except PermissionError:
    print("没有权限访问文件")
else:
    print("文件读取成功")
finally:
    print("操作完成")
```

### 函数参数示例
```python
def greet(name, greeting="Hello", *args, **kwargs):
    print(f"{greeting}, {name}!")
    if args:
        print("额外位置参数:", args)
    if kwargs:
        print("额外关键字参数:", kwargs)

greet("Alice")
greet("Bob", "Hi", "extra", age=25, city="Beijing")
```

## 常见问题

### Q: 什么时候使用列表推导式？
A: 列表推导式适合简单的列表生成，可以提高代码可读性和执行效率，但对于复杂逻辑建议使用常规循环。

### Q: 字典和集合的区别？
A: 字典是键值对映射，集合是无序的唯一元素集合。字典用于映射关系，集合用于去重和集合运算。

### Q: 异常处理的最佳实践是什么？
A: 尽量捕获具体的异常类型，使用with语句管理资源，在finally中进行清理工作，不要捕获所有异常。

---

**学习开始时间**：待定  
**预计完成时间**：待定


