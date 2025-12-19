# Matrix Copy

**难度**: Easy

**题目ID**: 31

**URL**: https://leetgpu.com/challenges/matrix-copy

---

Implement a program that copies an \(N \times N\) matrix of 32-bit floating point numbers from input array \(A\) to output array \(B\) on the GPU. The program should perform a direct element-wise copy so that \(B_{i,j} = A_{i,j}\) for all valid indices.

## Implementation Requirements

- External libraries are not permitted

  - The `solve` function signature must remain unchanged

  - The final result must be stored in matrix `B`

## Example 1:

```
Input:  A = [[1.0, 2.0],
             [3.0, 4.0]]
Output: B = [[1.0, 2.0],
             [3.0, 4.0]]
```

## Example 2:

```
Input:  A = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
Output: B = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
```

## Constraints

- 1 ≤ `N` ≤ 4096

  - All elements are 32-bit floating point numbers

---

*最后更新时间: 2025-12-18 21:34:24*
