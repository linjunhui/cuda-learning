# Ordinary Least Squares

**难度**: Medium

**题目ID**: 33

**URL**: https://leetgpu.com/challenges/ordinary-least-squares

---

Solve the Ordinary Least Squares (OLS) regression problem on a GPU. Given a feature matrix \(X\) of size \(n\_samples \times n\_features\) and a target vector \(y\) of size \(n\_samples\), compute the coefficient vector \(\beta\) that minimizes the sum of squared residuals:
\[ \min_{\beta} ||X\beta - y||^2 \]

The closed-form solution to OLS is:
\[ \beta = (X^TX)^{-1}X^Ty \]

## Implementation Requirements

- External libraries are not permitted.

  - The `solve` function signature must remain unchanged.

  - The final coefficients must be stored in the `beta` vector.

  - Assume that the feature matrix \(X\) is full rank (i.e., \(X^TX\) is invertible).

## Example:

Input:

\(X\) (samples × features): 
\[
\begin{bmatrix}
-0.23 & -0.23 & 1.52 \\
0.77 & -0.47 & 1.58 \\
-0.14 & 0.65 & 0.5 \\
-1.91 & -1.72 & 0.24 \\
-0.46 & -0.47 & 0.54
\end{bmatrix}
\]
\(y\): 
\[
\begin{bmatrix}
83.01 \\
93.4 \\
47.33 \\
-62.22 \\
13.06
\end{bmatrix}
\]
Output:

\(\beta\): 
\[
\begin{bmatrix}
13.97 \\
29.12 \\
61.05
\end{bmatrix}
\]

## Constraints

- 1 ≤ `n_samples` ≤ 100,000

  - 1 ≤ `n_features` ≤ 1,000

  - `n_samples` ≥ `n_features`

  - -1000.0 ≤ values in `X` and `y` ≤ 1000.0

  - Solutions are tested with absolute tolerance of 1e-2 and relative tolerance of 1e-2

---

*最后更新时间: 2025-12-18 21:34:24*
