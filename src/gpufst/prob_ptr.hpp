#ifndef GPUFST_GPUFST_GPU_PACK_HPP
#define GPUFST_GPUFST_GPU_PACK_HPP

#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>

//typedef uint64_t prob_ptr_t;
typedef unsigned long long int prob_ptr_t;
typedef float prob_ptr_float_t;

namespace gpufst{

__host__ __device__ prob_ptr_t pack (float prob, int ptr) {
  //assert (!isnan(prob));
  //assert (ptr >= 0 && ptr < 1L<<32);
  uint32_t i_prob = *(uint32_t *)&prob;
  if (i_prob & 0x80000000) 
    i_prob = i_prob ^ 0xFFFFFFFF;
  else
    i_prob = i_prob ^ 0x80000000;
  return (uint64_t)i_prob << 32 | ptr;
}

// Unpacks a probability.
__host__ __device__ float unpack_prob (prob_ptr_t packed) {
  uint32_t i_prob = packed >> 32;
  if (i_prob & 0x80000000) 
    i_prob = i_prob ^ 0x80000000;
  else
    i_prob = i_prob ^ 0xFFFFFFFF;
  return *(float *)&i_prob;
}

// Unpacks a back-pointer.
__host__ __device__ int unpack_ptr (prob_ptr_t packed) {
  //assert (!(packed & 0x80000000));
  return packed & 0x7FFFFFFF;
}

}

#endif
