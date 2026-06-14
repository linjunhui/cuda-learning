#pragma once
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      printf("CUDA error at %s:%d, code=%d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)
  