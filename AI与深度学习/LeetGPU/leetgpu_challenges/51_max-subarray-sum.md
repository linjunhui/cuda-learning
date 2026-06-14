# Max Subarray Sum

**难度**: Medium

**题目ID**: 51

**URL**: https://leetgpu.com/challenges/max-subarray-sum

---

Implement a program that computes the maximum sum of any contiguous subarray of length exactly `window_size`. You are given an array `input` of length `N` consisting of 32-bit signed integers, and an integer `window_size`.

## Implementation Requirements

- Use only native features (external libraries are not permitted)

    - The `solve` function signature must remain unchanged

    - The final result must be stored in the `output` variable

## Example 1:

```
Input:  input = [1, 2, 4, 2, 3], window_size = 2
Output: output = 6
```

## Example 2:

```
Input:  input = [-1, -4, -2, 1], window_size = 3
Output: output = -5
```

## Constraints

- 1 ≤ `N` ≤ 50,000

    - -10 ≤ `input[i]` ≤ 10

    - 1 ≤ `window_size` ≤ `N`

---

*最后更新时间: 2025-12-18 21:34:24*
