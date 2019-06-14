// tensor/cuda-utils.cc

// Copyright      2019  Johns Hopkins University (author: Daniel Povey)

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

#include "tensor/cuda-utils.h"
#include "base/kaldi-math.h"

namespace kaldi {
namespace tensor {

#define KALDI_STANDARD_THREAD_BLOCK_SIZE 256
#define KALDI_TARGET_NUM_THREAD_BLOCKS 1024


/**
   This function splits the kernel that's the last element of 'kernels' so that
   it satisifes grid_dim.x <= 65535-- if necessary, by splitting it into
   multiple kernels, increasing the length of the vector 'kernels'.
 */
static void SplitStandardKernelX(std::vector<StandardKernel> *kernels) {
  int cur_grid_dim = kernels->back().grid_dim.x;
  if (cur_grid_dim <= 65535)
    return;
  int num_kernels = (kernels->back().grid_dim.x + 65534) / 65535;

  size_t cur_size = kernels.size(),
      new_size = cur_size + num_kernels - 1;

  std::vector<int> new_grid_dims(num_kernels,
                                 cur_grid_dim / num_kernels);
  for (int i = 0; i < cur_grid_dim % num_kernels; i++)
    new_grid_dims[i]++;
  // the above ensures that the sum of new_grid_dims equals
  // cur_grid_dim; this is checked at the bottom of this function.
  StandardKernel prev_kernel = kernels->back();
  kernels->resize(new_size, prev_kernel);

  int prev_grid_dim_sum = 0;
  for (int i = 0; i < num_kernels; i++) {
    StandardKernel &k = (*kernels)[cur_size - 1 + i];
    int this_grid_dim = new_grid_dims[i];

    k.dim_grid.x = new_grid_dims[i];
    // If this is not the last i value (the last kernel), we can
    // leave k.sizes.max_offset_a.x as it is because we have
    // a 'whole number' of

    if (i + 1 < num_kernels) {
      // the following actually has no effect on operation, it's more
      // for clarity.
      k.sizes.max_offset_a.x = this_grid_dim * k.sizes.block_stride_a.x;
    } else {
      // for last one, this limit does make a difference, as the
      // highest-numbered thread block may not have all threads run.
      k.sizes.max_offset_a.x -= prev_grid_dim_sum * k.sizes.block_stride_a.x;
    }
    k.base_offset_a += prev_grid_dim_sum * k.sizes.block_stride_a.x;

    prev_grid_dim_sum += this_grid_dim;
  }
  KALDI_ASSERT(prev_grid_dim_sum == cur_grid_dim);
}

static void SplitStandardKernelY(std::vector<StandardKernel> *kernels) {
  // TODO.  Copy of the X one above.
}

static void GetStandardKernel1(const Pattern &a, const Pattern &b,
                               std::vector<StandardKernel> *kernels) {
  //  KALDI_PARANOID_ASSERT(a.num_axes == 1);

  // Note: the following call will invoke the constructor of dim3 which
  // sets all the values to 1, so we don't have to set the unused
  // gridDim elements.
  kernels->resize(kernels->size() + 1);
  StandardKernel &k = kernels->back();
  // Note: b.dims[0] is either 'dim' or 1; it won't affect anything, we only
  // need b's stride.
  int dim = a.dims[0],
      a_stride = a.strides[0],
      b_stride = b.strides[0];
  int bs = KALDI_STANDARD_THREAD_BLOCK_SIZE;
  int num_blocks = (dim + bs - 1) / bs;  // round up.

  k.sizes.thread_stride_a.x = a_stride;
  k.sizes.block_stride_a.x = a_stride * bs;

  k.sizes.thread_stride_b.x = b_stride;
  k.sizes.block_stride_b.x = b_stride * bs;

  k.sizes.max_offset_a.x = dim * a_stride;

  k.block_dim.x = std::min<int32>(bs, dim);
  k.grid_dim.x = num_blocks;

  if (num_blocks > 65535)
    SplitStandardKernelX(kernels);
}


// Fills out the 'x' dimension of the standard kernel using raxis 0
// of the patterns (which are assumed to have been sorted on the
// stride of a, so that raxis 0 is the one with the smallest stride,
// hopefully equal to 1)
//
// Does
static void GetStandardKernelX(const Pattern &a, const Pattern &b,
                               std::vector<StandardKernel> *kernels) {
  //  KALDI_PARANOID_ASSERT(a.num_axes == 1);

  // Note: the following call will invoke the constructor of dim3 which
  // sets all the values to 1, so we don't have to set the unused
  // gridDim elements.
  kernels->resize(kernels->size() + 1);
  StandardKernel &k = kernels->back();
  // Note: b.dims[0] is either 'dim' or 1; it won't affect anything, we only
  // need b's stride.
  int dim = a.dims[0],
      a_stride = a.strides[0],
      b_stride = b.strides[0];
  int bs = KALDI_STANDARD_THREAD_BLOCK_SIZE;
  int num_blocks = (dim + bs - 1) / bs;  // round up.

  k.sizes.thread_stride_a.x = a_stride;
  k.sizes.block_stride_a.x = a_stride * bs;

  k.sizes.thread_stride_b.x = b_stride;
  k.sizes.block_stride_b.x = b_stride * bs;

  k.sizes.max_offset_a.x = dim * a_stride;

  k.block_dim.x = std::min<int32>(bs, dim);
  k.grid_dim.x = num_blocks;

  if (num_blocks > 65535)
    SplitStandardKernelX(kernels);
}



static void GetStandardKernel2(const Pattern &a, const Pattern &b,
                               std::vector<StandardKernel> *kernels) {
  // Note: the following call will invoke the constructor of dim3 which
  // sets all the values to 1, so we don't have to set the unused
  // gridDim elements.
  kernels->resize(kernels->size() + 1);

  StandardKernel &k = kernels->back();
  int dim0 = a.dims[0],
      a_stride0 = a.strides[0],
      b_stride0 = b.strides[0],
      dim1 = a.dims[1],
      a_stride1 = a.strides[1],
      b_stride1 = b.strides[1];
  // We expect the patterns will have been normalized prior to this
  // call, which is why we don't expect zero strides for a.
  // some of the code does assume this, so we check for it.
  KALDI_PARANOID_ASSERT(a_stride0 != 0 && a_stride0 < a_stride1);

  if (dim0 < 64) {
    // dim0 is on the small side for a thread-block size, so we want the thread
    // block size to include part of dim1.
    int bs0 = dim0,
        dim0_rounded_up = RoundUpToNearestPowerOfTwo(dim0),
        bs1 = KALDI_STANDARD_THREAD_BLOCK_SIZE / dim0_rounded_up,
        nb1 = (dim1 + bs1 - 1) / bs1;
    k.block_dim.x = dim0;
    k.grid_dim.x = 1;  // it had this value anyway; this is for clrity.
    k.block_dim.y = bs1;
    k.grid_dim.y = nb1;

    k.sizes.max_offset_a.x = dim0 * a_stride0;
    k.sizes.max_offset_a.y = dim1 * a_stride1;

    k.sizes.thread_stride_a.x = a_stride0;
    k.sizes.block_stride_a.x = a_stride0 * bs0;
    k.sizes.thread_stride_a.y = a_stride1;
    k.sizes.block_stride_a.y = a_stride1 * bs1;

    k.sizes.thread_stride_b.x = b_stride0;
    k.sizes.block_stride_b.x = b_stride0 * bs0;
    k.sizes.thread_stride_b.y = b_stride1;
    k.sizes.block_stride_b.y = b_stride1 * bs1;

  } else {
    int bs0 = std::min<int32>(dim0, KALDI_STANDARD_THREAD_BLOCK_SIZE),
        nb0 = (dim0 + bs0 - 1) / bs0,
        bs1 = 1,
        nb1 = dim1;

    k.block_dim.x = dim0;
    k.grid_dim.x = 1;  // it had this value anyway; this is for clrity.
    k.block_dim.y = bs1;
    k.grid_dim.y = nb1;

    k.sizes.max_offset_a.x = dim0 * a_stride0;
    k.sizes.max_offset_a.y = dim1 * a_stride1;

    if (nb0 > 65535)
      SplitStandardKernelX(kernels);
    else if (nb1 > 65535)
      SplitStandardKernelY(kernels);
    // we don't handle the case where they are both > 65535, because that, times
    // the block size, would be more than the memory of any GPU, and would
    // require code changes.


  }


    // everything goes in the x, and we rely on the loop limits to
    //

  }

  if (dim0 * dim1 < 1024) {
    // Do it in a single thread block.  There's no point wasting
    // time figuring out more details.
  } else if (dim0 > bs && dim1 * dim2 <= 16384) {
    // 16384 is 4 * 4096, and 4096 is a kind of upper limit on
    // how many threads we might expect to run at once.

  }



  KALDI_PARANOID_ASSERT(dim0 > 1 && dim1 > 1);
  int bs = KALDI_STANDARD_THREAD_BLOCK_SIZE;
  if (dim0 >= bs)
  if (dim0 < bs) {
    if (dim0 >= bs / 2) {
      bs = dim0;
    } else {
      // This is a relatively complex case; the blocks can't just
      // be on dim0, they have to also include dim1.  We
      // would prefer to use an exact divisor of dim1, to avoid
      // having to use 2 kernels.
      int block_x = dim0,
          block_y = -1;
      float block_size_cost = 1.0e+10;
      if (dim0 * dim1 <= 1024) {
        block_y = dim1;
      } else {
        for (int this_block_y = 1;
             this_block_y * block_x < 1024;
             this_block_y++) {
          if (dim1 % this_block_y == 0) {
            int this_block_size = this_block_y * block_x;
            float this_block_size_cost =  GetBlockSizeCost(this_block_size);
            if (this_block_size_cost < block_size_cost) {
              block_size_cost = this_block_size_cost;
              block_y = this_block_y;
            }
          }
        }
      }
      if (this_block_y == -1) {
        block_y = KALDI_STANDARD_KERNEL1_BLOCK_SIZE / block_x;
        // and we'll deal with the remainder via a second kernel.
      }

      }

    }



  }

  int bs = KALDI_STANDARD_KERNEL1_BLOCK_SIZE;
  int32 num_blocks = (dim + bs - 1) / bs;  // round up.
  if (num_blocks > 1536)   // Don't want to have stragglers, so
    num_blocks = 1024;     // only limit num_blocks to 1024 if
                           // most will loop at least twice.
  k.sizes.thread_stride_a.x = a_stride;
  k.sizes.block_stride_a.x = a_stride * bs;
  k.sizes.thread_stride_b.x = b_stride;
  k.sizes.block_stride_b.x = b_stride * bs;
  k.block_dim.x = bs;
  k.grid_dim.x = num_blocks;
  // We don't treat the case where dim < bs separately (e.g. setting
  // k.block_dim.x = dim), I don't think it would make any real difference.
}



void GetStandardKernel(const Pattern &a, const Pattern &b,
                       std::vector<StandardKernel> *kernels) {

  // TODO: ensure that the 1st dim of a is the one with smallest stride.

  KALDI_PARANOID_ASSERT(DimsGeq(a, b) && a.num_axes >= b.num_axes &&
                        Broadcastable(a, b));
  int32 num_axes = a.num_axes;
  switch (num_axes) {
    case 1:
      GetStandardKernel1(a, b, kernels);
      return;
    case 2:
      GetStandardKernel2(a, b, kernels);
      return;
    case 3:
      GetStandardKernel3(a, b, kernels);
      return;
  }

}




}  // namespace kaldi
}  // namespace tensor
