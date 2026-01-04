"""
#### 题目1：变量和数据类型
**难度**：★☆☆☆☆

编写程序实现以下功能：
1. 声明不同类型的变量（int, float, str, bool）
2. 进行类型转换
3. 使用type()函数检查类型
4. 使用isinstance()函数进行类型检查

**要求**：
```python
# 你的代码实现
# 1. 声明变量
# 2. 类型转换示例
# 3. 类型检查示例

# 测试代码
if __name__ == "__main__":
    # 在这里测试你的代码
    pass
"""

if __name__ == "__main__":
    int_var = 10
    float_var = 1.0
    str_var = "hello"
    bool_var = True

    float_var2 = float(int_var)
    print(type(float_var2))

    print(isinstance(int_var, int))

    print(isinstance(float_var, (int, float)))