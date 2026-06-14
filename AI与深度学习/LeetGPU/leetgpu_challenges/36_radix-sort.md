# Radix Sort

**难度**: Medium

**题目ID**: 36

**URL**: https://leetgpu.com/challenges/radix-sort

---

Implement a radix sort algorithm that sorts an array of 32-bit unsigned integers on a GPU.
The program should take an input array of unsigned integers and sort them in ascending order using the radix sort algorithm.
The `input` parameter contains the unsorted array, and the sorted result should be stored in the `output` array.

## Implementation Requirements

- External libraries are not permitted

    - The `solve` function signature must remain unchanged

    - The final sorted result must be stored in the `output` array

    - Use radix sort algorithm (not other sorting algorithms)

    - Sort in ascending order

## Example 1:

```
Input:  [170, 45, 75, 90, 2, 802, 24, 66]
  Output: [2, 24, 45, 66, 75, 90, 170, 802]
```

## Example 2:

```
Input:  [1, 4, 1, 3, 555, 1000, 2]
  Output: [1, 1, 2, 3, 4, 555, 1000]
```

## Constraints

- `1 ≤ N ≤ 100,000,000`

    - `0 ≤ input[i] ≤ 4,294,967,295` (32-bit unsigned integers)

---

*最后更新时间: 2025-12-18 21:34:24*
