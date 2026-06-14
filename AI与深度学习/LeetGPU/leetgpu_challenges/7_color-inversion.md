# Color Inversion

**难度**: Easy

**题目ID**: 7

**URL**: https://leetgpu.com/challenges/color-inversion

---

Write a program to invert the colors of an image. The image is
represented as a 1D array of RGBA (Red, Green, Blue, Alpha) values, where each
component is an 8-bit unsigned integer (`unsigned char`).

Color inversion is performed by subtracting each color component (R, G, B)
from 255. The Alpha component should remain unchanged.

The input array
`image` will contain `width * height * 4` elements. The
first 4 elements represent the RGBA values of the top-left pixel, the next 4
elements represent the pixel to its right, and so on.

## Implementation Requirements

- Use only native features (external libraries are not permitted)

  - The `solve` function signature must remain unchanged

  - The final result must be stored in the array
`image`

## Example 1:

```
Input: image = [255, 0, 128, 255, 0, 255, 0, 255], width=1, height=2
Output: [0, 255, 127, 255, 255, 0, 255, 255]
```

## Example 2:

```
Input: image = [10, 20, 30, 255, 100, 150, 200, 255], width=2, height=1
Output: [245, 235, 225, 255, 155, 105, 55, 255]
```

## Constraints

- 1 ≤ `width` ≤ 4096

  - 1 ≤ `height` ≤ 4096

  - `width * height` ≤ 8,388,608.

---

*最后更新时间: 2025-12-18 21:34:24*
