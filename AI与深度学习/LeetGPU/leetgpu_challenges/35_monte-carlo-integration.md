# Monte Carlo Integration

**难度**: Medium

**题目ID**: 35

**URL**: https://leetgpu.com/challenges/monte-carlo-integration

---

Implement Monte Carlo integration on a GPU. Given a set of function values \(y_i = f(x_i)\) sampled at random points \(x_i\) uniformly distributed in the interval \([a, b]\), estimate the definite integral:
\[ \int_a^b f(x) \, dx \approx (b - a) \cdot \frac{1}{n} \sum_{i=1}^{n} y_i \]

The Monte Carlo method approximates the integral by computing the average of the function values and multiplying by the interval width.

## Implementation Requirements

- External libraries are not permitted

  - The `solve` function signature must remain unchanged

  - The final result must be stored in the `result` variable

  - Solutions are tested with absolute tolerance of 1e-2 and relative tolerance of 1e-2

## Example:

```
Input:  a = 0, b = 2, n_samples = 8
        y_samples = [0.0625, 0.25, 0.5625, 1.0, 1.5625, 2.25, 3.0625, 4.0]
Output: result = 3.1875
```

## Constraints

- 1 ≤ `n_samples` ≤ 100,000,000

  - -1000.0 ≤ `a` < `b` ≤ 1000.0

  - -10000.0 ≤ function values ≤ 10000.0

  - The tolerance is set to 1e-2 to account for the inherent randomness in Monte Carlo methods and floating-point precision variations.

---

*最后更新时间: 2025-12-18 21:34:24*
