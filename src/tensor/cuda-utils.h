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


/**
   These utilities are mostly for use with non-reducing (but possibly
   broadcasting) kernels.  The setup is: we have two Tensors a and b.  We are
   doing some operation like, say, a = sigmoid(b) that's non-reducing (no
   summation) but possibly broadcasting.

   For generality and also (reasonable) speed, we have a standard pattern/interface
   of kernel for such operations.


  void _standard_kernel(StandardTwoArgKernelSizes f, float *a, float *b) {
    int a_offset_x = f.thread_stride_a.x * threadIdx.x + block_stride_a.x * blockIdx.x,
      a_offset_yz = f.thread_stride_a.y * threadIdx.y + block_stride_a.y * blockIdx.y +
                    f.thread_stride_a.z * threadIdx.z + block_stride_a.z * blockIdx.z
    int b_offset_x = f.thread_stride_b.x * threadIdx.x + block_stride_b.x * blockIdx.x,
      b_offset_yz = f.thread_stride_b.y * threadIdx.y + block_stride_b.y * blockIdx.y +
                    f.thread_stride_b.z * threadIdx.z + block_stride_b.z * blockIdx.z

     for (; a_offset_x < f.max_offset_a;
        a_offset_x += block_stride_a * blockDim.x,
        b_offset_x += block_stride_b * blockDim.x) {
     a[a_offset_x + a_offset_yz] = some_func(b[b_offset_x + b_offset_yz]);
  }

  It's possible to encode a great variety of elementwise operations of up to 6
  dimensions using the pattern above; the rare cases that can't be handled that
  way can be handled using multiple invocations of the same kernel.

  We don't make any special allowances for things like matrix transpose, though;
  in future we may make a special variety of kernel that can handle transposes
  while using coalesced memory access.

  *Algorithm*.

  We first ensure that the first raxis (raxis=0) of pattern a has the smallest
  abs(stride).  This is necessary later in certain cases for the loop to work
  correctly.

  We try various algorithms for generating the kernel info; each one
  returns a score, and we then select the one that gave the best score.




  switch(num_axes) {
    case 0:

    case 1:
      pretty easy.
    case 2:
      copy our matrix code.
    case 3:



  }

  Simplest algorithm, applicable for up to 4 axes and if 1st axis is >= 256:




  : 1st axis gets allocated to
  block-dim x and spills over if necessary into grid-dim x.  Remaining dims go
  into grid-dim x if not used, then grid-dims y and z.

     Measure on:
        coalesced memory access (no, should always have this).

        - Loop length


        number of blocks; want no more than about 1024
        kernel size (prefer around 256; too small much worse than
         much worse).



  Next algorithm (only applicable if 1st dim is between 32 and 1024 and there are
  >=2 dims, and the product of the remaining dims is >1024:

  Make the loop be over one of those remaining dims.


  We assume raxis 0 of a has stride 1, which it will if any dim had
  stride 1.




  The loop with `a_offset_x < f.max_offset_a` allows us to cover several elements with
  one kernel (reducing kernel startup cost) and also makes it possible to fit




  Depends on the dim...
    Only one dim:
       Type 1 kernel using a while loop and the if-statement,
       with only the x dimension; thread block size = 128,
       number of blocks no greater than 1024.


   Two dims: # Note: we are assuming dim 0 is the one with stride=1 (if any).
     Make sure that, for a, the first axis dominates the second.  (c.f.
     axis-dominance property).

     If first dim >= 64  # first dim alone will be the thread dim.
       if (first_dim < 200) { // purposely between powers of 2.
          threadDim.x = first_dim;
          blockDim.x = 1
          threadDim.y = 1
          blockDim.y = second_dim;  # Use multiple kernels if limit of
                                    # 65536 is an issue.
       } else {
         # assume 2nd dim becomes blockDim.y; work out the max num-blocks
         # we might want of 1st dim.

         # break up first_dim into blocks of 128;
         # use the num-blocks given above if it's limiting,
         # and loop for the rest.
       }
    else (first_dim < 64),
       swap the x and y axes; use 256/first_dim to limit thread-block
       size.


    More than two dims (up to 5).
       Sort dims from smallest to greatest stride.

       First dim maps to 1st dimension.  If it is
       <128, we'll have to augment the thread block size
       with another dim.
         - First choice: find a dim whose product
           with the first dim is <1024, and if one
           exists, take the closest one to 256.
         - Second choice: take the next-smallest-stride
           dim, choose 256/first_dim as the thread
           block size, and put the rest of it in
           the grid size. [would go to x, while the
           1st choice goes to the y.]

       Now iterate through the




       If first dim is small, increase block size with another
       dim.  Choose smallest remaining dims as blockDim.y and blockDim.z,
       as long as num-blocks < 1024.


       (Choose one that gives num-blocks between 128
       and 1024 if already present; otherwise split one of the
       dims and put it as y).

       Put any remaining dims as gridDim.y and gridDim.z.
       If this isn't enough, use multiple kernel launches
       by (initially) iterating over the smallest dim.





   Sometimes we can handle something by launching two type 1 kernels, or
   a type 2 kernel

   Type 2 kernels use the x, y and z dimensions of
   grids and blocks


   First: we define type 1 kernel as a non-reducing (but possibly broadcasting)
   operation between two Tensors, e.g. a = b or a = sigmoid(b).  This is
   a rather general type of kernel that can be used as the generic case
   (applicable to arbitrary tensors).

   The KernelInfo is the part that needs to be passed into the
   kernel itself.  There are also two other things needed to launch the
   kernel:
<code>
      dim3 grid_dim, block_dim;
</code>

   The basic operation we'll do in the kernel is something like this;
   let 'a' and 'b' be pointers to float or something like that.  Let
<code>
    KernelInfo f;  // passed in.
    int x_offset_a = f.thread_stride_a.x * threadIdx.x + f.block_stride_a.x * blockIdx.x;
       y_offset_a = f.thread_stride_a.y * threadIdx.y + f.block_stride_a.y * blockIdx.y;
       z_offset_a = f.thread_stride_a.z * threadIdx.z + f.block_stride_a.z * blockIdx.z;
    // and similar statements to set x_offset_b, y_offset_b, z_offset_b.

    if (x_offset_a < f.max_offset_a.x &&
        y_offset_a < f.max_offset_a.y &&
        z_offset_a < f.max_offset_a.z)
      a[x_offset_a + y_offset_a + z_offset_a] =
          b[x_offset_b + y_offset_b + z_offset_b];


       thread_stride_a.y * threadIdx.y +
       thread_stride_a.z * threadIdx.z +
       block_stride_a.x * blockIdx.x
    // clock speed e.g. 3 gHz.  Say 100 instructions.
</code>
 */


struct StandardTwoArgKernelSizes {
  dim3 thread_stride_a;
  dim3 thread_stride_b;
  dim3 block_stride_a;
  dim3 block_stride_b;
  dim3 max_offset_a;
};




struct StandardTwoArgKernel {
  dim3 block_dim;
  dim3 grid_dim;
  StandardTwoArgKernelSizes sizes;  // passed into kernel.
};


/**
   This function returns the dimensions/sizes for one or more "standard kernels"
   to execute a "standard operation" on patterns a and b.  We define a
   standard operation as an elementwise operation possibly with broadcasting,
   of the form:
       a[i] = f(b[i])

   for some scalar function f, where i is an index-tuple.  a and b must be
   broadcastable, and the dims of a must be >= the corresponding dims of b
   (i.e.: no reduction).  We also require a.num_axes >= b.num_axes.
   The standard kernel is as follows:
<code>
  void _standard_kernel(StandardTwoArgKernelSizes f, float *a, float *b) {
    int a_offset_x = f.thread_stride_a.x * threadIdx.x + block_stride_a.x * blockIdx.x,
      a_offset_y = f.thread_stride_a.y * threadIdx.y + block_stride_a.y * blockIdx.y,
      a_offset_z = f.thread_stride_a.z * threadIdx.z + block_stride_a.z * blockIdx.z;
    int b_offset = f.thread_stride_b.x * threadIdx.x + block_stride_b.x * blockIdx.x +
                   f.thread_stride_b.y * threadIdx.y + block_stride_b.y * blockIdx.y +
                   f.thread_stride_b.z * threadIdx.z + block_stride_b.z * blockIdx.z

     if (a_offset_x < f.max_offset_a.x && a_offset_y < f.max_offset_a.y)
       a[a_offset_x + a_offset_y + a_offset_z] = some_func(b[b_offset]);
  }
</code>
  }

      @param [in] a   First pattern for which we want the kernel (or kernels)
      @param [in] b   Second pattern for which we want the kernel (or kernels)
      @param [out] kernels  The kernels are *appended to* this vector (this
                      allows for recursive operation in this function).  Normally,
                      we'll have `kernels->size() == 1` at exit.  The user is expected
                      to call all of them (the order doesn't matter, and they
                      don't have to be called in sequence).
 */
void GetStandardKernel(const Pattern &a, const Pattern &b,
                       std::vector<StandardKernel> *kernels);



/**
   First: we define type 1 kernel as a non-reducing (but possibly broadcasting)
   operation between two Tensors, e.g. a = b or a = sigmoid(b).  This is
   a rather general type of kernel that can be used as the generic case
   (applicable to arbitrary tensors).

   The KernelInfo is the part that needs to be passed into the
   kernel itself.  There are also two other things needed to launch the
   kernel:
<code>
      dim3 grid_dim, block_dim;
</code>

   The basic operation we'll do in the kernel is something like this;
   let 'a' and 'b' be pointers to float or something like that.  Let
<code>
    Type1KernelInfo f;  // passed in.
    int x_offset_a = f.thread_stride_a.x * threadIdx.x + f.block_stride_a.x * blockIdx.x;
       y_offset_a = f.thread_stride_a.y * threadIdx.y + f.block_stride_a.y * blockIdx.y;
       z_offset_a = f.thread_stride_a.z * threadIdx.z + f.block_stride_a.z * blockIdx.z;
    // and similar statements to set x_offset_b, y_offset_b, z_offset_b.

    if (x_offset_a < f.max_offset_a.x &&
        y_offset_a < f.max_offset_a.y)
      a[x_offset_a + y_offset_a + z_offset_a] =
          b[x_offset_b + y_offset_b + z_offset_b];
    // clock speed e.g. 3 gHz.  Say 100 instructions.
</code>
 */

class StandardKernelSizes {
  dim3 thread_stride_a;
  dim3 thread_stride_b;
  dim3 block_stride_a;
  dim3 block_stride_b;
  dim3 max_offset_a;
};

class StandardKernel {
  dim3 dim_block;
  dim3 dim_grid;
  StandardKernelSizes sizes;
  // offset_a and offset_b are offsets that we have to add to the data-pointers
  // of a and b before we call the kernel; these will normally be zero, but may
  // be nonzero if we have to generate multiple kernels due to, say, size
  // constraints.
  int64 base_offset_a{0};
  int64 base_offset_b{0};
};




}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_PATTERN_H_
