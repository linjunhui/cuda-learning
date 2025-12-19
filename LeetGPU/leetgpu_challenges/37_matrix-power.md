# Matrix Power

**难度**: Medium

**题目ID**: 37

**URL**: https://leetgpu.com/challenges/matrix-power

---

Implement a GPU program that raises a square matrix \(A\) of size \(N \times N\) to an integer power \(P\).

The `solve` function receives a flattened input matrix `input` (row-major order), an empty output matrix `output` of the same size, the dimension `N`, and the exponent `P`.

You must compute \(\text{output} = A^{P}\) where matrix multiplication is standard dense multiplication over 32-bit floating point numbers.

## Implementation Requirements

- External libraries are **not** permitted.

    - The `solve` function signature must remain unchanged.

    - The final result must be written to the `output` array in row-major order.

## Example 1:

```
Input:
    input  = [[1.0, 2.0],
              [3.0, 4.0]]
    N      = 2
    P      = 3
  Output:
    output = [[37.0, 54.0],
              [81.0, 118.0]]
```

## Example 2:

```
Input:
    input  = [[1.0, 0.0, 2.0],
              [0.0, 1.0, 0.0],
              [3.0, 0.0, 0.0]]
    N      = 3
    P      = 2
  Output:
    output = [[7.0, 0.0, 2.0],
              [0.0, 1.0, 0.0],
              [3.0, 0.0, 6.0]]
```

## Constraints

- \(1 \le N \le 1024\)

    - \(1 \le P \le 20\)

    - Elements of `input` satisfy \(-10.0 \le A_{ij} \le 10.0\)

---

*最后更新时间: 2025-12-18 21:34:24*
