# Python面向对象学习

## 学习目标

深入理解Python面向对象编程，掌握类设计、继承、多态等核心概念，能够使用设计模式解决实际问题。

## 学习时间

1个月（30天）

## 学习内容

### 第1周：类和对象基础
- 类和对象概念
  - 类的定义语法
  - 对象的创建和使用
  - 实例变量和类变量
  - 实例方法和类方法
  - 静态方法（@staticmethod）
  - 类方法（@classmethod）
- 封装
  - 私有属性和方法（名称改写）
  - 属性访问控制
  - 属性的getter和setter
  - 属性装饰器（@property）
- 构造函数和析构函数
  - __init__方法
  - __new__方法
  - __del__方法
  - 对象初始化流程

### 第2周：继承和多态
- 继承
  - 单继承和多继承
  - 方法重写（override）
  - 方法解析顺序（MRO）
  - super()函数
  - 抽象基类（ABC）
  - 接口设计
- 多态
  - 多态概念
  - 鸭子类型（Duck Typing）
  - 多态实现
- 组合和聚合
  - 组合关系
  - 聚合关系
  - 组合vs继承

### 第3周：特殊方法
- 对象表示
  - __str__和__repr__
  - __format__
  - __bytes__
- 比较运算符
  - __eq__, __ne__
  - __lt__, __le__, __gt__, __ge__
  - __hash__
- 数值运算
  - 算术运算符重载（__add__, __sub__, __mul__等）
  - 反向运算符（__radd__等）
  - 增量赋值运算符（__iadd__等）
  - 一元运算符（__neg__, __pos__等）
- 容器操作
  - __len__
  - __getitem__, __setitem__, __delitem__
  - __contains__
  - __iter__, __next__
- 属性访问
  - __getattr__, __setattr__, __delattr__
  - __getattribute__
  - __dir__
- 调用和上下文
  - __call__
  - __enter__, __exit__
  - __copy__, __deepcopy__

### 第4周：设计模式和高级特性
- 属性访问和描述符
  - 描述符协议（__get__, __set__, __delete__）
  - 数据描述符和非数据描述符
  - 描述符应用场景
- 抽象基类
  - abc模块
  - 抽象方法（@abstractmethod）
  - 抽象属性（@abstractproperty）
- 设计模式（Python实现）
  - 创建型模式
    - 单例模式（Singleton）
    - 工厂模式（Factory）
    - 建造者模式（Builder）
  - 结构型模式
    - 适配器模式（Adapter）
    - 装饰器模式（Decorator）
    - 外观模式（Facade）
  - 行为型模式
    - 观察者模式（Observer）
    - 策略模式（Strategy）
    - 模板方法模式（Template Method）
- SOLID原则
  - 单一职责原则（SRP）
  - 开闭原则（OCP）
  - 里氏替换原则（LSP）
  - 接口隔离原则（ISP）
  - 依赖倒置原则（DIP）

## 实践项目

### 项目1：类设计实践（第1周）
设计并实现几个类：
- 银行账户类
- 学生信息管理类
- 使用属性装饰器管理属性

### 项目2：继承体系设计（第2周）
设计一个继承体系：
- 图形类层次结构（Shape, Circle, Rectangle等）
- 使用抽象基类定义接口
- 实现多态

### 项目3：特殊方法应用（第3周）
实现一个自定义类：
- 使用特殊方法实现容器行为
- 实现运算符重载
- 实现上下文管理器

### 项目4：设计模式实践（第4周）
实现几个设计模式：
- 实现观察者模式
- 实现策略模式
- 应用设计模式解决实际问题

## 学习资源

### 书籍
- 《流畅的Python》（第8-11章）
- 《Effective Python》（第22-30条）
- 《设计模式：Python语言实现》

### 在线资源
- Python官方文档（数据模型）
- Real Python OOP教程
- Design Patterns in Python

## 每日学习计划

### 工作日（2-3小时）
- 理论学习：1小时
- 编程实践：1-2小时

### 周末（4-6小时）
- 理论学习：2小时
- 编程实践：2-4小时

## 检查点

### 第1周检查点
- [ ] 理解类和对象的概念
- [ ] 掌握封装和属性访问控制
- [ ] 能够设计简单的类
- [ ] 完成项目1

### 第2周检查点
- [ ] 理解继承和多态
- [ ] 掌握MRO和super()的使用
- [ ] 理解抽象基类
- [ ] 完成项目2

### 第3周检查点
- [ ] 掌握常用特殊方法
- [ ] 能够实现运算符重载
- [ ] 理解描述符协议
- [ ] 完成项目3

### 第4周检查点
- [ ] 理解并实现设计模式
- [ ] 掌握SOLID原则
- [ ] 能够进行面向对象设计
- [ ] 完成项目4

## 代码示例

### 属性装饰器示例
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("半径不能为负数")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2
```

### 抽象基类示例
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)
```

### 描述符示例
```python
class PositiveNumber:
    def __init__(self):
        self.name = None
    
    def __get__(self, instance, owner):
        return instance.__dict__[self.name]
    
    def __set__(self, instance, value):
        if value < 0:
            raise ValueError("值必须为正数")
        instance.__dict__[self.name] = value
    
    def __set_name__(self, owner, name):
        self.name = name

class BankAccount:
    balance = PositiveNumber()
    
    def __init__(self, balance):
        self.balance = balance
```

## 常见问题

### Q: Python的私有属性是如何实现的？
A: Python使用名称改写（name mangling）实现私有属性，在属性名前加双下划线（如__private），Python会在内部将其改写为_ClassName__private。

### Q: 什么时候使用继承，什么时候使用组合？
A: 使用继承表示"是一个"关系，使用组合表示"有一个"关系。优先使用组合，只有在需要多态行为时才使用继承。

### Q: MRO是什么？如何查看MRO？
A: MRO（Method Resolution Order）是方法解析顺序，决定在多继承中方法查找的顺序。可以使用类名.__mro__或类名.mro()查看。

---

**学习开始时间**：待定  
**预计完成时间**：待定




