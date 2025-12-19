# Rainbow Table

**难度**: Easy

**题目ID**: 24

**URL**: https://leetgpu.com/challenges/rainbow-table

---

Implement a program that performs `R` rounds of parallel hashing on an array of 32-bit integers using the provided hash function.
The hash should be applied `R` times iteratively (the output of one round becomes the input to the next).

## Implementation Requirements

- External libraries are not permitted

    - The `solve` function signature must remain unchanged

    - The final result must be stored in array `output`

## Example 1:

```
Input:  numbers = [123, 456, 789], R = 2
Output: hashes = [1636807824, 1273011621, 2193987222]
```

## Example 2:

```
Input:  numbers = [0, 1, 2147483647], R = 3
Output: hashes = [96754810, 3571711400, 2006156166]
```

## Constraints

- 1 ≤ `N` ≤ 10,000,000

    - 1 ≤ `R` ≤ 100

    - 0 ≤ `input[i]` ≤ 2147483647

---

*最后更新时间: 2025-12-18 21:34:24*
