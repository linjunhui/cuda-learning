#!/usr/bin/env python3
"""
Python类型注解语法演示
展示与C++尾置返回类型相似的语法
"""

from typing import TypeVar, Union, List, Dict, Any
import math

# 类型变量定义
T = TypeVar('T')
U = TypeVar('U')

# 1. 基本函数类型注解
def add(a: int, b: int) -> int:
    """简单的加法函数"""
    return a + b

def multiply(a: float, b: float) -> float:
    """乘法函数"""
    return a * b

# 2. 联合类型 (Python 3.10+)
def process_number(num: int | float) -> int | float:
    """处理数字，支持int或float"""
    return num * 2

# 3. 泛型函数
def combine(a: T, b: T) -> T:
    """泛型函数，类型由参数决定"""
    return a + b

# 4. 复杂返回类型
def get_statistics(numbers: List[float]) -> Dict[str, float]:
    """返回统计信息"""
    if not numbers:
        return {"mean": 0.0, "std": 0.0}
    
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    std = math.sqrt(variance)
    
    return {
        "mean": mean,
        "std": std,
        "min": min(numbers),
        "max": max(numbers)
    }

# 5. 可选返回类型
def find_element(lst: List[T], target: T) -> T | None:
    """查找元素，可能返回None"""
    for item in lst:
        if item == target:
            return item
    return None

# 6. 函数类型注解
def apply_operation(a: float, b: float, op: callable) -> float:
    """应用操作函数"""
    return op(a, b)

# 7. 变量类型注解
def demonstrate_variable_annotations():
    """演示变量类型注解"""
    # 基本类型注解
    name: str = "Python类型注解"
    age: int = 25
    height: float = 175.5
    is_student: bool = True
    
    # 容器类型注解
    numbers: List[int] = [1, 2, 3, 4, 5]
    scores: Dict[str, float] = {"math": 95.5, "english": 88.0}
    
    # 联合类型
    value: int | float = 42
    value = 3.14  # 可以重新赋值为float
    
    print(f"姓名: {name}")
    print(f"年龄: {age}")
    print(f"身高: {height}")
    print(f"是学生: {is_student}")
    print(f"数字列表: {numbers}")
    print(f"成绩: {scores}")
    print(f"值: {value}")

def main():
    """主函数演示各种类型注解用法"""
    print("=== Python类型注解语法演示 ===\n")
    
    # 1. 基本函数调用
    print("1. 基本函数调用:")
    result1 = add(5, 3)
    result2 = multiply(3.14, 2.86)
    print(f"add(5, 3) = {result1}")
    print(f"multiply(3.14, 2.86) = {result2}")
    print()
    
    # 2. 联合类型
    print("2. 联合类型:")
    num1 = process_number(10)
    num2 = process_number(3.14)
    print(f"process_number(10) = {num1}")
    print(f"process_number(3.14) = {num2}")
    print()
    
    # 3. 泛型函数
    print("3. 泛型函数:")
    int_result = combine(5, 3)
    float_result = combine(3.14, 2.86)
    str_result = combine("Hello", " World")
    print(f"combine(5, 3) = {int_result}")
    print(f"combine(3.14, 2.86) = {float_result}")
    print(f"combine('Hello', ' World') = {str_result}")
    print()
    
    # 4. 复杂返回类型
    print("4. 复杂返回类型:")
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = get_statistics(data)
    print(f"数据: {data}")
    print(f"统计信息: {stats}")
    print()
    
    # 5. 可选返回类型
    print("5. 可选返回类型:")
    numbers = [1, 2, 3, 4, 5]
    found = find_element(numbers, 3)
    not_found = find_element(numbers, 10)
    print(f"在{numbers}中查找3: {found}")
    print(f"在{numbers}中查找10: {not_found}")
    print()
    
    # 6. 函数类型
    print("6. 函数类型:")
    result = apply_operation(5.0, 3.0, lambda x, y: x + y)
    print(f"apply_operation(5.0, 3.0, +) = {result}")
    print()
    
    # 7. 变量类型注解
    print("7. 变量类型注解:")
    demonstrate_variable_annotations()
    print()
    
    print("=== 与C++尾置返回类型的对比 ===")
    print("Python: def function(param: type) -> return_type:")
    print("C++:    auto function(param) -> return_type")
    print()
    print("相似点:")
    print("- 都使用 '->' 表示返回类型")
    print("- 都支持类型推导")
    print("- 都支持泛型编程")
    print()
    print("差异点:")
    print("- Python: 运行时类型检查，C++: 编译时类型检查")
    print("- Python: 动态类型，C++: 静态类型")
    print("- Python: 参数类型在参数名后，C++: 参数类型在参数名前")

if __name__ == "__main__":
    main()






















