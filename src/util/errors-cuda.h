#ifndef ERRORS_CUDA_H_
#define ERRORS_CUDA_H_

#include <cstdio>

static void _handleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s:%d: %s\n", file, line, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

#define handleError(err) (_handleError(err, __FILE__, __LINE__))

#endif