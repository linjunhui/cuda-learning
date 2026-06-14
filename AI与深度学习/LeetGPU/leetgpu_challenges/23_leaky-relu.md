# Leaky ReLU

**难度**: Easy

**题目ID**: 23

**URL**: https://leetgpu.com/challenges/leaky-relu

---

Implement a program that performs the leaky ReLU activation function on a vector of floating-point numbers. The leaky ReLU function is defined as:
\[ f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases} \]
where \(\alpha\) is a small positive constant (0.01 in this problem).

## Implementation Requirements

- External libraries are not permitted

    - The `solve` function signature must remain unchanged

    - The final result must be stored in vector `output`

    - Use \(\alpha = 0.01\) as the leaky coefficient

## Example 1:

```
Input:  x = [1.0, -2.0, 3.0, -4.0]
  Output: y = [1.0, -0.02, 3.0, -0.04]
```

## Example 2:

```
Input:  x = [-1.5, 0.0, 2.5, -3.0]
  Output: y = [-0.015, 0.0, 2.5, -0.03]
```

## Constraints

- 1 ≤ `N` ≤ 100,000,000

    - -1000.0 ≤ `input[i]` ≤ 1000.0

---

*最后更新时间: 2025-12-18 21:34:24*
