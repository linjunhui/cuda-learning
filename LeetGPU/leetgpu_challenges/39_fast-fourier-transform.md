# Fast Fourier Transform

**难度**: Hard

**题目ID**: 39

**URL**: https://leetgpu.com/challenges/fast-fourier-transform

---

Implement a GPU program that computes the Fast Fourier Transform (FFT) of a
complex-valued 1-D signal. Given an input `signal` array containing
`N` complex numbers stored as interleaved real/imaginary pairs,
compute the discrete Fourier transform and store the result in the
`spectrum` array. The FFT converts a time-domain signal into its
frequency-domain representation using the formula: \[ X_k = \sum_{n=0}^{N-1}
x_n \cdot e^{-j 2\pi kn / N} \quad \text{for } k = 0, 1, \ldots, N-1 \] The
FFT algorithm reduces the computational complexity from O(N²) to O(N log N) by
exploiting symmetries in the twiddle factors.

## Implementation Requirements

- External libraries (cuFFT etc.) are not permitted

  - The `solve` function signature must remain unchanged

  - The final result must be stored in the `spectrum` array

  - The kernel must be entirely GPU-resident—no host-side FFT calls

  - Both input and output use interleaved real/imaginary layout:
`[real₀, imag₀, real₁, imag₁, ...]`

## Example 1:

```
Input:  N = 4
        signal = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        (represents: [1+0j, 0+0j, 0+0j, 0+0j])

Output: spectrum = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        (represents: [1+0j, 1+0j, 1+0j, 1+0j])
```

## Example 2:

```
Input:  N = 2
        signal = [1.0, 0.0, 1.0, 0.0]
        (represents: [1+0j, 1+0j])

Output: spectrum = [2.0, 0.0, 0.0, 0.0]
        (represents: [2+0j, 0+0j])
```

## Constraints

- `1 ≤ N ≤ 262,144`

  - All values are 32-bit floating point numbers

  - Absolute error ≤ 1e-3 and relative error ≤ 1e-3

  - Input and output arrays have length `2 × N`

---

*最后更新时间: 2025-12-18 21:34:24*
