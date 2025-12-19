# Prefix Sum

**难度**: Medium

**题目ID**: 16

**URL**: https://leetgpu.com/challenges/prefix-sum

---

Write a GPU program that computes the prefix sum (cumulative sum) of an array of 32-bit floating point numbers.
For an input array `[a, b, c, d, ...]`, the prefix sum is `[a, a+b, a+b+c, a+b+c+d, ...]`.

## Implementation Requirements

- Use only GPU native features (external libraries are not permitted)

  - The `solve` function signature must remain unchanged

  - The result must be stored in the `output` array

## Example 1:

```
Input: [1.0, 2.0, 3.0, 4.0]
Output: [1.0, 3.0, 6.0, 10.0]
```

## Example 2:

```
Input: [5.0, -2.0, 3.0, 1.0, -4.0]
Output: [5.0, 3.0, 6.0, 7.0, 3.0]
```

## Constraints

- 1 ≤ `N` ≤ 100,000,000

  - -1000.0 ≤ `input[i]` ≤ 1000.0

  - The largest value in the output array will fit within a 32-bit float

---

*最后更新时间: 2025-12-18 21:34:24*
