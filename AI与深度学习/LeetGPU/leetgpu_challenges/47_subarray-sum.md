# Subarray Sum

**难度**: Medium

**题目ID**: 47

**URL**: https://leetgpu.com/challenges/subarray-sum

---

Implement a program that computes the sum of a subarray of 32-bit integers.
You are given an input array `input` of length `N`, and two indices `S` and `E`.
`S` and `E` are inclusive, 0-based start and end indices — compute the sum of `input[S..E]`.

## Implementation Requirements

- Use only native features (external libraries are not permitted)

  - The `solve` function signature must remain unchanged

  - The final result must be stored in the `output` variable

## Example 1:

```
Input: input = [1, 2, 1, 3, 4], S = 1, E = 3
Output: output = 6
```

## Example 2:

```
Input: input = [1, 2, 3, 4], S = 0, E = 3
Output: output = 10
```

## Constraints

- 1 ≤ `N` ≤ 100,000,000

  - 1 ≤ `input[i]` ≤ 10

  - 0 ≤ `S` ≤ `E` ≤ `N - 1`

---

*最后更新时间: 2025-12-18 21:34:24*
