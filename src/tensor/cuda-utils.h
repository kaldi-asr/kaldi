// tensor/cuda-utils.h

//  Copyright      2019  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_TENSOR_TENSOR_CUDA_UTILS_H_
#define KALDI_TENSOR_TENSOR_CUDA_UTILS_H_ 1

// Caution: don't include this header if we're not compiling with cuda.

#include "tensor/tensor-common.h"
#include <cuda_runtime_api.h>
#include <limits>


namespace kaldi {
namespace tensor {



struct StandardOneArgKernelSizes {
  dim3 thread_stride_a;
  dim3 block_stride_a;
  dim3 mindex_a_range;
};


class StandardOneArgKernel {
  dim3 dim_block;
  dim3 dim_grid;
  StandardOneArgKernelSizes sizes;
  // offset_a is an offset that we have to add to the data-pointer of a before
  // we call the kernel; it will normally equal the 'offset' members of the
  // pattern, but may be different if we have to generate multiple kernels due
  // to, say, size constraints.
  int64 offset_a;
};


/**
   This function returns the dimensions/sizes for one or more "standard one arg
   kernels" to execute a "standard one arg operation" on Tensors a, of which
   only the patterns are provided.  We define a standard one-arg operation as an
   in-place elementwise operation of the form:

       a[i] = f(a[i])

   where i is an index-tuple in the index-tuple-set of the pattern a; search in
   pattern.h for the meaning of this notation.  Note: one-arg kernels may not
   actually be needed in practice as two-arg kernels with a and b identical
   can do the same thing.

   The standard one-arg kernel is as follows:
<code>
template <typename T>
__global__ void _some_one_arg_kernel(StandardOneArgKernelSizes f, T *a) {
    int a_offset_x = f.thread_stride_a.x * threadIdx.x + block_stride_a.x * blockIdx.x,
      a_offset_y = f.thread_stride_a.y * threadIdx.y + block_stride_a.y * blockIdx.y,
      a_offset_z = f.thread_stride_a.z * threadIdx.z + block_stride_a.z * blockIdx.z,
      a_offset = a_offset_x + a_offset_y + a_offset_z;

     if (a_offset_x < f.mindex_a_range.x && a_offset_y < f.mindex_a_range.y)
       a[a_offset] = some_func(a[a_offset]);
  }

  // which would be invoked as follows:
  template <typename T>
  void some_one_arg_kernel(const Tensor &a,
                               const StandardOneArgKernel &k) {
    _some_one_arg_kernel<<<k.grid_dim, k.block_dim>>>(
         k.sizes(), a.GetData<T>() + k.base_offset_a);
         b.GetData<T>() + k.base_offset_b);
  }
  //

</code>
  }

      @param [in] a   Fattern for which we want the kernel (or kernels).  This is
                      an elementwise operation so it must be in-place.
                      All its strides are required to be positive (hence it may
                      have not trivial axes).
      @param [out] kernels  The kernels are output to this vector.  Normally,
                      we'll have `kernels->size() == 1` at exit.  The user is expected
                      to call all of them (the order doesn't matter).
 */
void GetStandardOneArgKernel(const Pattern &a,
                             std::vector<StandardOneArgKernel> *kernels);




struct StandardTwoArgKernelSizes {
  dim3 thread_stride_a;
  dim3 thread_stride_b;
  dim3 block_stride_a;
  dim3 block_stride_b;
  dim3 mindex_a_range;
};


class StandardTwoArgKernel {
  dim3 dim_block;
  dim3 dim_grid;
  StandardTwoArgKernelSizes sizes;
  // offset_a and offset_b are offsets that we have to add to the data-pointers
  // of a and b before we call the kernel; these will normally equal the
  // 'offset' members of the respective patterns, but they may be different from
  // those if we have to generate multiple kernels due to, say, size
  // constraints.
  int64 offset_a;
  int64 offset_b;
};


/**
   This function returns the dimensions/sizes for one or more "standard two arg
   kernels" to execute a "standard two arg operation" on Tensors a and b, of
   which only the patterns are provided.  We define a standard two-arg operation
   as an elementwise operation possibly with broadcasting, of the form:

       a[i] = f(b[i])
   where i is an index-tuple in the index-tuple-set of the pattern-tuple (a,b);
   search in pattern.h for the meaning of this notation.

   a and b must be broadcastable, and the dims of a must be >= the corresponding
   dims of b (i.e.: no reduction).  We also require a.num_axes >= b.num_axes,
   which results from the tuple (a,b) having been reduced (see ReducePatternTuple()
   in pattern-tuple-utils.h).
   The standard two-arg kernel is as follows:
<code>
template <typename T>
__global__ void _some_two_arg_kernel(StandardTwoArgKernelSizes f, T *a, const T *b) {
    int a_offset_x = f.thread_stride_a.x * threadIdx.x + block_stride_a.x * blockIdx.x,
      a_offset_y = f.thread_stride_a.y * threadIdx.y + block_stride_a.y * blockIdx.y,
      a_offset_z = f.thread_stride_a.z * threadIdx.z + block_stride_a.z * blockIdx.z;
    int b_offset = f.thread_stride_b.x * threadIdx.x + block_stride_b.x * blockIdx.x +
                   f.thread_stride_b.y * threadIdx.y + block_stride_b.y * blockIdx.y +
                   f.thread_stride_b.z * threadIdx.z + block_stride_b.z * blockIdx.z;

     if (a_offset_x < f.mindex_a_range.x && a_offset_y < f.mindex_a_range.y)
       a[a_offset_x + a_offset_y + a_offset_z] = some_func(b[b_offset]);
  }

  // which would be invoked as follows:
  template <typename T>
  void some_two_arg_kernel(const Tensor &a, const Tensor &b,
                               const StandardTwoArgKernel &k) {
    _some_two_arg_kernel<<<k.grid_dim, k.block_dim>>>(
         k.sizes,
         a.GetData<T>() + k.base_offset_a,
         b.GetData<T>() + k.base_offset_b);
  }

</code>
  There is also way to invoke two-arg kernels "in-place" so that the function
  takes two args, like a = f(a, b).

      @param [in] a   First pattern for which we want the kernel (or kernels).
                      All its strides are required to be positive (hence it may
                      have not trivial axes).
      @param [in] b   Second pattern for which we want the kernel (or kernels)
                      Must satisfy Broadcastable(a, b).
      @param [out] kernels  The kernels are output this vector.  Normally,
                      we'll have `kernels->size() == 1` at exit.  The user is expected
                      to call all of them (the order doesn't matter).
 */
void GetStandardTwoArgKernel(const Pattern &a, const Pattern &b,
                             std::vector<StandardTwoArgKernel> *kernels);




class StandardThreeArgKernelSizes {
  dim3 thread_stride_a;
  dim3 thread_stride_b;
  dim3 thread_stride_c;

  dim3 block_stride_a;
  dim3 block_stride_b;
  dim3 block_stride_c;

  dim3 mindex_a_range;
};

class StandardThreeArgKernel {
  dim3 dim_block;
  dim3 dim_grid;
  StandardTwoArgKernelSizes sizes;

  // base_offset_{a,b,c} are offsets that we have to add to the data-pointers of
  // the storage regions of a, b and c before we call the kernel; these will
  // normally equal the 'offset' members of the input patterns, but they may
  // differ from those if we have to generate multiple kernels due to, say, size
  // constraints.
  int64 base_offset_a;
  int64 base_offset_b;
  int64 base_offset_c;
};


/**
   This function returns the dimensions/sizes for one or more "standard three arg
   kernels" to execute a "standard three arg operation" on Tensors a and b, of
   which only the patterns are provided.  We define a standard three-arg operation
   as an elementwise operation possibly with broadcasting, of the form:

       a[i] = f(b[i], c[i])
   where i is an index-tuple in the index-tuple-set of the pattern-tuple (a,b,c);
   search in pattern.h for the meaning of this notation.

   a, b and c must be broadcastable, and the dims of a must be >= the
   corresponding dims of b and of c (i.e.: no reduction).  We also require
   a.num_axes >= b.num_axes and a.num_aces >= c.num_axes, which results from the
   tuple (a,b,c) having been reduced (see ReducePatternTuple() in
   pattern-tuple-utils.h).

   The standard three-arg kernel is as follows:
<code>
template <typename T>
  void _some_three_arg_kernel(StandardThreeArgKernelSizes f,
                              T *a, const T *b, const T *c) {
    int a_offset_x = f.thread_stride_a.x * threadIdx.x + block_stride_a.x * blockIdx.x,
      a_offset_y = f.thread_stride_a.y * threadIdx.y + block_stride_a.y * blockIdx.y,
      a_offset_z = f.thread_stride_a.z * threadIdx.z + block_stride_a.z * blockIdx.z;
    int b_offset = f.thread_stride_b.x * threadIdx.x + block_stride_b.x * blockIdx.x +
                   f.thread_stride_b.y * threadIdx.y + block_stride_b.y * blockIdx.y +
                   f.thread_stride_b.z * threadIdx.z + block_stride_b.z * blockIdx.z,
        c_offset = f.thread_stride_c.x * threadIdx.x + block_stride_c.x * blockIdx.x +
                   f.thread_stride_c.y * threadIdx.y + block_stride_c.y * blockIdx.y +
                   f.thread_stride_c.z * threadIdx.z + block_stride_c.z * blockIdx.z;

     if (a_offset_x < f.mindex_a_range.x && a_offset_y < f.mindex_a_range.y)
       a[a_offset_x + a_offset_y + a_offset_z] = some_func(b[b_offset], c[c_offset]);
  }

  // which would be invoked as follows:
  template <typename T>
  void some_three_arg_kernel(const Tensor &a, const Tensor &b,
                                 const Tensor &c,
                                 const StandardThreeArgKernel &k) {
    _some_three_arg_kernel<<<k.grid_dim, k.block_dim>>>(
         k.sizes,
         a.GetData<T>() + k.base_offset_a,
         b.GetData<T>() + k.base_offset_b,
         c.GetData<T>() + k.base_offset_c);
  }
</code>
  }

      @param [in] a   First pattern for which we want the kernel (or kernels).
                      All its strides are required to be positive (hence it may
                      have not trivial axes).
      @param [in] b   Second pattern for which we want the kernel (or kernels)
      @param [in] c   Third pattern for which we want the kernel (or kernels)
      @param [out] kernels  The kernels are output this vector.  Normally,
                      we'll have `kernels->size() == 1` at exit.  The user is expected
                      to call all of them (the order doesn't matter).
 */
void GetStandardThreeArgKernel(const Pattern &a, const Pattern &b,
                               std::vector<StandardThreeArgKernel> *kernels);





}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_PATTERN_H_
