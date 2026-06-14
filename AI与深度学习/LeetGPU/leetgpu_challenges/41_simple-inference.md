# Simple Inference

**难度**: Easy

**题目ID**: 41

**URL**: https://leetgpu.com/challenges/simple-inference

---

Run inference on a PyTorch model. Given an input tensor and a trained `torch.nn.Linear` model, compute the forward pass and store the result in the output tensor.

The model performs a linear transformation: `output = input @ weight.T + bias` where `weight` has shape [output_size, input_size] and `bias` has shape [output_size].

## Implementation Requirements

- Use PyTorch's built-in functions and operations

    - The `solve` function signature must remain unchanged

    - The final result must be stored in the `output` tensor

    - The model is already loaded and ready for inference

## Example 1:

```
Input:  input = [[1.0, 2.0]]  (batch_size=1, input_size=2)
          model: Linear layer with weight=[[0.5, 1.0], [1.5, 0.5]], bias=[0.1, 0.2]
  Output: output = [[2.6, 2.7]]  (batch_size=1, output_size=2)
```

## Example 2:

```
Input:  input = [[1.0], [2.0], [3.0]]  (batch_size=3, input_size=1)
          model: Linear layer with weight=[[2.0]], bias=[1.0]
  Output: output = [[3.0], [5.0], [7.0]]  (batch_size=3, output_size=1)
```

## Constraints

- 1 ≤ `batch_size` ≤ 1,000

    - 1 ≤ `input_size` ≤ 1,000

    - 1 ≤ `output_size` ≤ 1,000

    - -10.0 ≤ input values ≤ 10.0

---

*最后更新时间: 2025-12-18 21:34:24*
