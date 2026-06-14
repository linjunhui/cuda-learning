# Sigmoid Linear Unit

**难度**: Easy

**题目ID**: 52

**URL**: https://leetgpu.com/challenges/sigmoid-linear-unit

---

Implement the SiLU (Sigmoid Linear Unit) activation function forward pass for 1D input vectors. 
Given an input tensor of shape [N] where N is the number of elements, compute the output using the elementwise formula.

SiLU is defined as:
\[
\begin{align}
\sigma(x) &= \frac{1}{1 + e^{-x}} \\
\text{SiLU}(x) &= x \cdot \sigma(x)
\end{align}
\]

## Implementation Requirements

- Use only native features (external libraries are not permitted)

  - The `solve` function signature must remain unchanged

  - The final result must be stored in the `output` tensor

## Example 1:

```
Input:  input = [0.5, 1.0, -0.5]  (N=3)
Output: output = [0.3112295, 0.731059, -0.1887705]
```

## Example 2:

```
Input:  input = [-1.0, -2.0, -3.0, -4.0, -5.0]  (N=5)
Output: output = [-0.26894143 -0.23840584 -0.14227763 -0.07194484 -0.03346425]
```

## Constraints

- 1 ≤ `N` ≤ 10,000

  - -100.0 ≤ input values ≤ 100.0

---

*最后更新时间: 2025-12-18 21:34:24*
