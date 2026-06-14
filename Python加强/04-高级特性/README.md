# Python高级特性学习

## 学习目标

掌握Python高级特性，包括装饰器、生成器、元类等，能够编写更加优雅和高效的Python代码。

## 学习时间

1个月（30天）

## 学习内容

### 第1周：装饰器
- 函数装饰器基础
  - 装饰器概念和作用
  - 函数装饰器的实现
  - 装饰器语法糖（@）
  - 装饰器的执行顺序
- 带参数的装饰器
  - 装饰器工厂函数
  - 三层嵌套函数
  - 装饰器参数传递
- 类装饰器
  - 类装饰器的实现
  - 类装饰器的应用
  - __call__方法的使用
- 装饰器的应用
  - 日志记录装饰器
  - 性能计时装饰器
  - 权限检查装饰器
  - 缓存装饰器
  - 重试装饰器
- 内置装饰器
  - @property
  - @staticmethod
  - @classmethod
  - @functools.wraps
  - @functools.lru_cache

### 第2周：生成器和迭代器
- 迭代器协议
  - 迭代器概念
  - __iter__和__next__方法
  - 迭代器的实现
  - 迭代器的应用
- 生成器基础
  - 生成器概念
  - yield关键字
  - 生成器函数
  - 生成器表达式
  - 生成器vs迭代器
- 生成器进阶
  - 生成器的方法（send, throw, close）
  - 协程和yield from
  - 生成器链
  - 生成器应用场景
- 迭代工具
  - itertools模块
  - 常用迭代器函数（chain, cycle, groupby等）
  - 生成器表达式的高级用法

### 第3周：上下文管理器和元编程基础
- 上下文管理器
  - 上下文管理器协议（__enter__, __exit__）
  - with语句的工作原理
  - 上下文管理器的应用
  - contextlib模块
  - @contextmanager装饰器
  - 嵌套上下文管理器
- 属性访问控制
  - __getattr__和__setattr__
  - __getattribute__
  - property和描述符回顾
- 描述符协议深入
  - 描述符协议详解
  - 数据描述符和非数据描述符
  - 描述符的应用场景
  - 描述符实现缓存

### 第4周：元类和元编程
- 类对象和实例对象
  - type类的作用
  - 类的创建过程
  - 元类的概念
- 元类基础
  - 元类的定义
  - 元类的使用
  - 元类的执行时机
  - __new__和__init__在元类中的作用
- 元类的应用
  - 自动注册子类
  - 单例模式实现
  - ORM框架应用
  - API验证
- 元编程技巧
  - 动态创建类
  - 动态修改类
  - 类的动态属性
  - 代码生成

## 实践项目

### 项目1：装饰器工具库（第1周）
实现常用装饰器：
- 实现日志记录装饰器
- 实现性能计时装饰器
- 实现缓存装饰器
- 实现重试装饰器

### 项目2：生成器应用（第2周）
使用生成器解决实际问题：
- 实现文件读取生成器
- 实现数据管道处理
- 实现无限序列生成器
- 优化内存使用

### 项目3：上下文管理器应用（第3周）
实现自定义上下文管理器：
- 实现数据库连接上下文管理器
- 实现文件操作上下文管理器
- 实现锁的上下文管理器
- 应用contextlib模块

### 项目4：元类应用（第4周）
使用元类解决实际问题：
- 实现自动注册框架
- 实现ORM基础框架
- 实现API验证框架

## 学习资源

### 书籍
- 《流畅的Python》（第7、9、14、21章）
- 《Effective Python》（第31-50条）
- 《Python Tricks》

### 在线资源
- Python官方文档（数据模型、装饰器）
- Real Python高级教程
- Python Descriptor Guide

## 每日学习计划

### 工作日（2-3小时）
- 理论学习：1小时
- 编程实践：1-2小时

### 周末（4-6小时）
- 理论学习：2小时
- 编程实践：2-4小时

## 检查点

### 第1周检查点
- [ ] 理解装饰器的原理和实现
- [ ] 能够编写函数装饰器和类装饰器
- [ ] 掌握装饰器的常见应用
- [ ] 完成项目1

### 第2周检查点
- [ ] 理解迭代器和生成器
- [ ] 能够编写生成器函数
- [ ] 掌握生成器的应用场景
- [ ] 完成项目2

### 第3周检查点
- [ ] 理解上下文管理器协议
- [ ] 能够实现自定义上下文管理器
- [ ] 掌握描述符协议
- [ ] 完成项目3

### 第4周检查点
- [ ] 理解元类的概念和原理
- [ ] 能够使用元类解决问题
- [ ] 掌握元编程技巧
- [ ] 完成项目4

## 代码示例

### 装饰器示例
```python
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end - start:.4f}秒")
        return result
    return wrapper

def cache(func):
    cache_dict = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args in cache_dict:
            return cache_dict[args]
        result = func(*args)
        cache_dict[args] = result
        return result
    return wrapper

@timer
@cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

### 生成器示例
```python
def fibonacci_generator():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def read_file_generator(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield line.strip()
```

### 上下文管理器示例
```python
from contextlib import contextmanager

@contextmanager
def file_handler(filename, mode):
    f = open(filename, mode)
    try:
        yield f
    finally:
        f.close()

class DatabaseConnection:
    def __enter__(self):
        self.conn = connect_to_database()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
        return False
```

### 元类示例
```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    pass
```

## 常见问题

### Q: 装饰器和函数包装的区别？
A: 装饰器是函数包装的语法糖，使用@decorator语法更加简洁。装饰器本质上是一个返回函数的函数。

### Q: 什么时候使用生成器？
A: 当需要处理大量数据但不需要全部加载到内存时，使用生成器。生成器可以节省内存并提高性能。

### Q: 元类什么时候使用？
A: 元类主要用于框架开发，如ORM、API框架等。普通应用开发中很少需要直接使用元类。

### Q: 描述符和属性的区别？
A: @property装饰器是描述符的简化版本。描述符更灵活，可以在多个类之间复用，而属性只适用于单个类。

---

**学习开始时间**：待定  
**预计完成时间**：待定



