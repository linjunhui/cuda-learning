# Histogramming

**难度**: Medium

**题目ID**: 13

**URL**: https://leetgpu.com/challenges/histogramming

---

Write a GPU program that computes the histogram of an array of 32-bit integers.
The histogram should count the number of occurrences of each integer value in the range `[0, num_bins)`.
You are given an input array `input` of length `N` and the number of bins `num_bins`.

The result should be an array of integers of length
`num_bins`, where each element represents
the count of occurrences of its corresponding index in the input array.

## Implementation Requirements

- Use only native features (external libraries are not permitted)

  - The
`solve` function signature must remain unchanged

  - The final result must be stored in the
`histogram` array.

## Examples

```
Input: input = [0, 1, 2, 1, 0],  N = 5, num_bins = 3
Output: [2, 2, 1]
  
Input: input = [3, 3, 3, 3], N = 4, num_bins = 5
Output: [0, 0, 0, 4, 0]
```

## Constraints

- 1 ≤ `N` ≤ 100,000,000

  - 0 ≤ `input[i]` < `num_bins`

  - 1 ≤ `num_bins` ≤ 1024

---

*最后更新时间: 2025-12-18 21:34:24*
