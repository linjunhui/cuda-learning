# 1D Convolution

**难度**: Easy

**题目ID**: 9

**URL**: https://leetgpu.com/challenges/1d-convolution

---

Implement a program that performs a 1D convolution operation. Given an input array and a kernel (filter), compute the convolved
output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied
where it fully overlaps with the input.

The input consists of two arrays:

- `input`: A 1D array of 32-bit floating-point numbers.

  - `kernel`: A 1D array of 32-bit floating-point numbers representing the convolution kernel.

The output should be written to the `output` array, which will have a size of `input_size - kernel_size + 1`.

The convolution operation is defined mathematically as:

\[
output[i] = \sum_{j=0}^{kernel\_size-1} input[i + j] \cdot kernel[j]
\]

where \(i\) ranges from 0 to \(input\_size - kernel\_size\).

## Implementation Requirements

- Use only native features (external libraries are not permitted)

  - The
`solve` function signature must remain unchanged

  - The final result must be stored in the array
`output`

## Example 1:

```
Input: input = [1, 2, 3, 4, 5], kernel = [1, 0, -1]
Output: [-2, -2, -2]
```

## Example 2:

```
Input: input = [2, 4, 6, 8], kernel = [0.5, 0.2]
Output: [1.8, 3.2, 4.6]
```

## Constraints

- 1 ≤ `input_size` ≤ 1,500,000

  - 1 ≤ `kernel_size` ≤ 2047

  - `kernel_size` ≤ `input_size`

---

*最后更新时间: 2025-12-18 21:34:24*
