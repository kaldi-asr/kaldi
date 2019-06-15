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
       @param [in] kernel  The input kernel that needs to be split;
                           must satisfy kernel.grid_dim.x > 65535.
       @param [out] kernels  The split copies of the input kernel will be
                          *appended* to the vector `kernels`.

 */
static void SplitStandardKernelX(const StandardThreeArgKernel &kernel,
                                 std::vector<StandardThreeArgKernel> *kernels) {
  int cur_grid_dim = kernels->back().grid_dim.x;
  KALDI_ASSERT(cur_grid_dim > 65535);
    return;
  int num_kernels = (kernels->back().grid_dim.x + 65534) / 65535;

  size_t cur_size = kernels.size(),
      new_size = cur_size + num_kernels - 1;

  std::vector<int> new_grid_dims(num_kernels,
                                 cur_grid_dim / num_kernels);
  // the next loop ensures that the sum of new_grid_dims equals cur_grid_dim,
  // correcting for the rounding down.  this will be checked at the bottom of
  // this function.
  for (int i = 0; i < cur_grid_dim % num_kernels; i++)
    new_grid_dims[i]++;


  int prev_grid_dim_sum = 0;
  for (int i = 0; i < num_kernels; i++) {
    kernels->push_back(kernel);
    StandardKernel &new_kernel = kernels->back();
    int this_grid_dim = new_grid_dims[i];

    new_kernel.dim_grid.x = this_grid_dim;
    if (i + 1 < num_kernels) {
      // the following actually has no effect on operation since all
      // threads will run; it's more for clarity.
      k.sizes.max_offset_a.x = this_grid_dim * k.sizes.block_stride_a.x;
    } else {
      // for the last kernel, this limit might actually make a difference, as
      // the highest-numbered thread block in the last kernel may not have all
      // threads run.
      k.sizes.max_offset_a.x -= prev_grid_dim_sum * k.sizes.block_stride_a.x;
    }
    k.base_offset_a += prev_grid_dim_sum * k.sizes.block_stride_a.x;
    k.base_offset_b += prev_grid_dim_sum * k.sizes.block_stride_b.x;
    k.base_offset_c += prev_grid_dim_sum * k.sizes.block_stride_c.x;

    prev_grid_dim_sum += this_grid_dim;
  }
  KALDI_ASSERT(prev_grid_dim_sum == cur_grid_dim);
}

// This is a copy of SplitStandardKernelX above, but with x's changed to y's.
// See the documentation for SplitStandardKernelX.
static void SplitStandardKernelY(const StandardThreeArgKernel &kernel,
                                 std::vector<StandardThreeArgKernel> *kernels) {
  int cur_grid_dim = kernels->back().grid_dim.y;
  KALDI_ASSERT(cur_grid_dim > 65535);
    return;
  int num_kernels = (kernels->back().grid_dim.y + 65534) / 65535;

  size_t cur_size = kernels.size(),
      new_size = cur_size + num_kernels - 1;

  std::vector<int> new_grid_dims(num_kernels,
                                 cur_grid_dim / num_kernels);
  // the next loop ensures that the sum of new_grid_dims equals cur_grid_dim,
  // correcting for the rounding down.  this will be checked at the bottom of
  // this function.
  for (int i = 0; i < cur_grid_dim % num_kernels; i++)
    new_grid_dims[i]++;


  int prev_grid_dim_sum = 0;
  for (int i = 0; i < num_kernels; i++) {
    kernels->push_back(kernel);
    StandardKernel &new_kernel = kernels->back();
    int this_grid_dim = new_grid_dims[i];

    new_kernel.dim_grid.y = this_grid_dim;
    if (i + 1 < num_kernels) {
      // the following actually has no effect on operation since all
      // threads will run; it's more for clarity.
      k.sizes.max_offset_a.y = this_grid_dim * k.sizes.block_stride_a.y;
    } else {
      // for the last kernel, this limit might actually make a difference, as
      // the highest-numbered thread block in the last kernel may not have all
      // threads run.
      k.sizes.max_offset_a.y -= prev_grid_dim_sum * k.sizes.block_stride_a.y;
    }
    k.base_offset_a += prev_grid_dim_sum * k.sizes.block_stride_a.y;
    k.base_offset_b += prev_grid_dim_sum * k.sizes.block_stride_b.y;
    k.base_offset_c += prev_grid_dim_sum * k.sizes.block_stride_c.y;

    prev_grid_dim_sum += this_grid_dim;
  }
  KALDI_ASSERT(prev_grid_dim_sum == cur_grid_dim);
}


/**
   This function is used to handle cases where we still have more than 3 axes
   (should be very rare since we only use the standard kernel on reduced
   pattern-tuples).  It creates copies of a kernel that differ only
   in max_offset_a, max_offset_b, max_offset_c, to take account of
   an raxis that has not been included in the kernel.

     @param [in] a      The first Pattern that's an arg to the kernel
     @param [in] b      The second Pattern that's an arg to the kernel
     @param [in] c      The third Pattern that's an arg to the kernel
     @param [in] raxis  The raxis that we're splitting on; in place of the
                        single input 'kernel' we will have a separate
                        output for each i in [0, a.dim[raxis] - 1]
     @param [in] kernel    The original kernel that awe are going to
                        expand.  Assumed to correspond to an index
                        value of 0 on raxis 'raxis'.
     @param [out] kernels  The output kernels are *appended* to this
                        vector.  The number of output kernels will
                        be a.dims[raxis].
 */
static void SplitStandardKernelByAxis(
    const Pattern &a,
    const Pattern &b,
    const Pattern &c,
    int32 raxis,
    const StandardThreeArgKernel &kernel
    std::vector<StandardThreeArgKernel> *kernels) {
  // Asserting raxis > 0 is just from knowledge of how the calling code works,
  // it is not something that would affect the operation of this function.
  KALDI_ASSERT(raxis > 0 && raxis < a.num_axes);
  int32 dim = a.dims[raxis];
  for (int32 i = 0; i < dim; i++) {
    kernels->push_back(kernel);
    StandardThreeArgKernel &k = kernels->back();
    k.max_offset_a += i * a.strides[raxis];
    k.max_offset_b += i * b.strides[raxis];
    k.max_offset_c += i * c.strides[raxis];
  }
}

// Fills out the 'x' dimension of the standard kernel, which is assumed to have
// immediately before been initialized with its default constructor.

// The 'x' dimension is filled out using raxis=0, which is required to be the
// lowest abs(stride) in 'a' and have stride != 0; most of the time, this stride
// will be 1.  We preferentially make the thread block vary along this axis,
// which will increase the chance of consolidated memory accesses.  (We could,
// of course, take much more care to ensure memory accesses are consolidated,
// taking into account the patterns of b and c and taking into account whether
// the start of the tensor is on a 128-byte boundary; we can consider these
// kinds of optimizations in future).
static void ProcessStandardKernelX(const Pattern &a,
                                   const Pattern &b,
                                   const Pattern &c,
                                   StandardThreeArgKernel *k) {
  KALDI_PARANOID_ASSERT(a.num_axes >= 1 && a.dims[0] > 1);
  // Note: b.dims[0] is either 'dim' or 1; it won't affect anything, we only
  // need b's stride.
  int dim = a.dims[0],
      a_stride = a.strides[0],
      b_stride = b.strides[0],
      c_stride = c.strides[0];

  // bs is the thread-block size (at least, as far as the x dimension is
  // concerned).
  int bs = std::min<int32>(RoundUpToNearestPowerOfTwo(dim),
                           KALDI_STANDARD_THREAD_BLOCK_SIZE);
  int num_blocks = (dim + bs - 1) / bs;  // round up.

  k->sizes.thread_stride_a.x = a_stride;
  k->sizes.block_stride_a.x = a_stride * bs;
  k->sizes.thread_stride_b.x = b_stride;
  k->sizes.block_stride_b.x = b_stride * bs;
  k->sizes.thread_stride_c.x = c_stride;
  k->sizes.block_stride_c.x = c_stride * bs;

  k->sizes.max_offset_a.x = dim * a_stride;

  k->block_dim.x = bs;
  k->grid_dim.x = num_blocks;
}


// Fills out the 'y' dimension of the standard three-arg kernel (whose x
// dimension is assumed to already have been set up) using an raxis-index
// specified by the user; this will normally be the one with the largest dim,
// and it won't be 0 because axis 0 goes to x and will already have been
// processed.
static void ProcessStandardKernelY(const Pattern &a,
                                   const Pattern &b,
                                   const Pattern &c,
                                   int32 raxis,
                                   StandardThreeArgKernel *kernel) {
  KALDI_PARANOID_ASSERT(a.num_axes > raxis && raxis > 0);

  int dim = a.dims[raxis],
      a_stride = a.strides[raxis],
      b_stride = b.strides[raxis],
      c_stride = c.strides[raxis];

  // bs means block size.
  int bs_x = kernel->block_dim.x;
  // If the threads-per-block is too small, we may have to have threads-per-block
  // != 1 on this axis.
  int bs_y = std::min<int32>(RoundUpToNearestPowerOfTwo(dim),
                             KALDI_STANDARD_THREAD_BLOCK_SIZE / bs_x);
  if (bs_y < 1)
    bs_y = 1;  // just for robustness to any later code changes.
  int num_blocks = (dim + bs_y - 1) / bs_y;  // round up.

  k->sizes.thread_stride_a.y = a_stride;
  k->sizes.block_stride_a.y = a_stride * bs_y;
  k->sizes.thread_stride_b.y = b_stride;
  k->sizes.block_stride_b.y = b_stride * bs_y;
  k->sizes.thread_stride_c.y = c_stride;
  k->sizes.block_stride_c.y = c_stride * bs_y;

  k->sizes.max_offset_a.y = dim * a_stride;
  k->block_dim.y = bs_y;
  k->grid_dim.y = num_blocks;
}


// Fills out the 'z' dimension of the standard kernel (whose x and y dimensions
// are assumed to already have been set up) using an raxis-index specified by the
// user; this will normally be the one with the largest dim, and it won't be 0
// because axis 0 goes to x and will already have been processed.
static void ProcessStandardKernelZ(const Pattern &a, const Pattern &b,
                                   int32 raxis,
                                   StandardThreeArgKernel *kernel) {
  KALDI_PARANOID_ASSERT(a.num_axes > raxis && raxis > 0);

  int dim = a.dims[raxis],
      a_stride = a.strides[raxis],
      b_stride = b.strides[raxis],
      c_stride = c.strides[raxis];

  // bs means block size.
  int bs_x = kernel->block_dim.x,
      bs_y = kernel->block_dim.y;
  // If the threads-per-block is too small, we may have to have grid_dim.z
  // != 1.  But this is only possible if we can choose a value of grid_dim.z
  // that exactly divides 'dim', because the kernel doesn't have an
  // if-statement for the z dimension.

  int bs_z;
  for (int i = 1; i * bs_x * bs_y <= KALDI_STANDARD_THREAD_BLOCK_SIZE; i++)
    if (dim % i == 0)
      bs_z = i;
  // Note: in the normal case, bs_z will be one now.  In all cases,
  // bs_z will divide 'dim' exactly.

  int num_blocks = dim / bs_z;  // round up.

  k->sizes.thread_stride_a.z = a_stride;
  k->sizes.block_stride_a.z = a_stride * bs_z;
  k->sizes.thread_stride_b.z = b_stride;
  k->sizes.block_stride_b.z = b_stride * bs_z;
  k->sizes.thread_stride_c.z = c_stride;
  k->sizes.block_stride_c.z = c_stride * bs_z;

  // The kernel code will not actually inspect max_offset_a.z; we just leave it
  // as a guide in case of future code changes.
  k->sizes.max_offset_a.z = dim * a_stride;

  k->block_dim.z = bs_z;
  k->grid_dim.z = num_blocks;
}




void FinalizeKernel(const Pattern &a,
                    const Pattern &b,
                    const Pattern &c,
                    ArrayRef<int32> remaining_axes,
                    std::vector<StandardThreeArgKernel> *kernels) {
  // prev_size is the size of 'kernels'  before the most recent one
  // was added (since GetStandardKernel appends).  Would normally be zero.
  size_t prev_size = kernels->size() - 1;
  if (kernels->back().grid_dim.x > 65535) {
    SplitStandardKernelX(kernels);
    if (kernels->back().grid_dim.y > 65535)
      KALDI_ERR << "You are trying to process a tensor that's way too big";
    // We don't handle the case where the x and y grid dims are both >65535,
    // because that much data wouldn't fit on the GPU anyway once you take into
    // account the thread block size.  (It would require code changes to do
    // correctly).
  } else if (kernels->back().grid_dim.y > 65535) {
    SplitStandardKernelY(kernels);
  }
  if (kernels->back().grid_dim.z > 65535)
    KALDI_ERR << "You are trying to process a tensor that's way too big";

  for (size_t i = 0; i < remaining_axes.size(); i++) {
    int32 raxis = remaining_axes[i];
    std::vector<StandardKernel> next_kernels;
    for (auto kernel: *kernels)
      SplitStandardKernelByAxis(a, b, c, raxis, kernel, next_kernels);
    kernels->swap(next_kernels);
  }
}


// Returns the raxis with the smallest abs(stride).  It is an error if any axis
// has stride = 0 (i.e. is a trivial axis).  Intended to be called
// from GetStandardKernel()
int32 RaxisWithSmallestAbsStride(const Pattern &p) {
  int32 num_axes = a.num_axes,
      ans = 0;
  for (int32 raxis = 1; raxis < num_axes; raxis++)
    if (abs(p.strides[raxis]) < abs(p.strides[ans]))
      ans = raxis;
  KALDI_ASSERT(p.strides[ans] != 0 &&
               "Args to GetStandardKernel() do not have the expected "
               "properties");
  // if the assert fails, either the pattern-tuple was not in reduced form, or
  // there is reduction in the operation, which is not allowed in a "standard"
  // kernel.
  return ans;
}


void GetStandardThreeArgKernel(const Pattern &a,
                               const Pattern &b,
                               const Pattern &c,
                               std::vector<StandardThreeArgKernel> *kernels) {
  KALDI_PARANOID_ASSERT(DimsGeq(a, b) && a.num_axes >= b.num_axes &&
                        Broadcastable(a, b));
  int32 smallest_stride_raxis = RaxisWithSmallestStride(a);
  if (smallest_stride_raxis != 0) {
    // This is unexpected but we can deal with it by swapping axes.
    Pattern a_new(a), b_new(b), c_new(c);
    TransposeR(0, smallest_stride_raxis, &a_new);
    TransposeR(0, smallest_stride_raxis, &b_new, true);
    TransposeR(0, smallest_stride_raxis, &c_new, true);
    GetStandardKernel(a_new, b_new, c_new, kernels);
    return;
  }
  kernels->clear();
  kernels->resize(1);
  Kernel *kernel = &(kernels->back());

  int32 num_axes = a.num_axes;
  switch (num_axes) {
    case 0:
      // The default constructor gives values suitable for a kernel that
      // only processes a single element, so there is nothing more to do.
    return;
    case 1:
      ProcessStandardKernelX(a, b, kernel);
      FinalizeKernel(a, b, {}, kernels);
      return;
    case 2:
      ProcessStandardKernelX(a, b, kernel);
      ProcessStandardKernelY(a, b, 1, kernel);
      FinalizeKernel(a, b, {}, kernels);
      return;
    default: {  // >= 3 axes
      ProcessStandardKernelX(a, b, kernel);
      // Sort the raxes 1, 2,... from greatest to least dimension.  (Note: there
      // are cases where this won't be optimal and we may want to take the
      // stride into account in order to ensure more consolidated memory access;
      // we could think about that later).
      std::vector<int32> raxes;
      for (int i = 1; i < num_axes; i++)
        raxes.push_back(i);
      std::sort(raxes.begin(), raxes.end(),
                // below is a C++11 lambda used as a comparator function, like
                // the operator x < y.  The "a" in brackets is the Pattern a,
                // declared above, which is a "captured" variable for this
                // lambda.
                [a] (int x, int y) {
                  // reverse the direction of comparison because we want raxes
                  // sorted from greatest to least dim.
                  return a.dims[x] > a.dims[y];
                });
      ProcessStandardKernelY(a, b, raxes[0], kernel);
      ProcessStandardKernelZ(a, b, raxes[1], kernel);
      raxes_data = &(raxes[0]);
      // The expression {raxes_data + 2, raxes_data + num_axes - 1} is a
      // constructor to ArrayRef which gives an array of ints including raxes[2]
      // and any remaining elements.  This is the possibly-empty subset of raxes
      // that we haven't already processed, and they should all have fairly
      // small dimension as we've sorted `raxes` from greatest to least
      // dimension.  We'll process these left-over raxes by duplicating the
      // kernel, shifting the base_offset_{a,b,c} value as needed.
      FinalizeKernel(a, b, {raxes_data + raxes_data + num_axes - 1},
                     raxes.begin  kernel);
      return;
    }
  }
}

// Convert from 3-arg to 2-arg kernel, discarding information.
static void ConvertKernel(const StandardThreeArgKernel &src,
                          StandardTwoArgKernel *dest) {
  dest->dim_block = src.dim_block;
  dest->dim_grid = src.dim_grid;
  dest->sizes.thread_stride_a = src.sizes.thread_stride_a;
  dest->sizes.thread_stride_b = src.sizes.thread_stride_b;
  dest->sizes.block_stride_a = src.sizes.block_stride_a;
  dest->sizes.block_stride_b = src.sizes.block_stride_b;
  dest->sizes.max_offset_a = src.sizes.max_offset_a;
  dest->base_offset_a = src.base_offset_a;
  dest->base_offset_b = src.base_offset_b;
}

// Convert from 3-arg to 1-arg kernel, discarding information.
static void ConvertKernel(const StandardThreeArgKernel &src,
                          StandardTwoArgKernel *dest) {
  dest->dim_block = src.dim_block;
  dest->dim_grid = src.dim_grid;
  dest->sizes.thread_stride_a = src.sizes.thread_stride_a;
  dest->sizes.block_stride_a = src.sizes.block_stride_a;
  dest->sizes.max_offset_a = src.sizes.max_offset_a;
  dest->base_offset_a = src.base_offset_a;
}

// Doing a 2-arg kernel by first doing the 3-arg one is of course
// wasteful
void GetStandardTwoArgKernel(const Pattern &a,
                             const Pattern &b,
                             const Pattern &c,
                             std::vector<StandardThreeArgKernel> *kernels) {
  std::vector<StandardThreeArgKernel> temp_kernels;
  GetStandardThreeArgKernel(a, b, b, &temp_kernels);
  size_t size = temp_kernels.size();
  kernels->resize(size);
  for (size_t i = 0; i < size; i++)
    ConvertKernel(temp_kernels[i],

}



}  // namespace kaldi
}  // namespace tensor
