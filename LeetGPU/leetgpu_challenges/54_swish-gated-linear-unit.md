# Swish-Gated Linear Unit

**难度**: Easy

**题目ID**: 54

**URL**: https://leetgpu.com/challenges/swish-gated-linear-unit

---

Implement the Swish-Gated Linear Unit (SWiGLU) activation function forward pass for 1D input vectors. 
Given an input tensor of shape [N] where N is the number of elements, compute the output using the elementwise formula.
The input and output tensor must be of type `float32`.

SWiGLU is defined as:

- Split input \(x\) into two halves: \(x_1\) and \(x_2\)

    - Compute SiLU on the first half:
\[
\text{SiLU}(x_1) = x_1 \cdot \sigma(x_1), \quad 
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

    - Compute the SWiGLU output:
\[
\text{SWiGLU}(x_1, x_2) = \text{SiLU}(x_1) \cdot x_2
\]

## Implementation Requirements

- Use only native features (external libraries are not permitted)

  - The `solve` function signature must remain unchanged

  - The final result must be stored in the `output` tensor

## Example 1:

```
Input:  [1.0, 2.0, 3.0, 4.0]  (N=4)
Output: [2.1931758, 7.0463767]
```

## Example 2:

```
Input:  [0.5, 1.0]  (N=2)
Output: [0.31122968]
```

## Constraints

- 1 ≤ `N` ≤ 100,000

  - N is an even number

  - -100.0 ≤ input values ≤ 100.0

---

*最后更新时间: 2025-12-18 21:34:24*
