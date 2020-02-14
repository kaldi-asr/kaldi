// cudadecoder/batched-static-nnet3-kernels.h
//
// Copyright (c) 2019; NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun
//
// Licensed under the Apache License; Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing; software
// distributed under the License is distributed on an "AS IS" BASIS;
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND; either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if HAVE_CUDA == 1

#include <cuda_runtime_api.h>
#include "base/kaldi-types.h"

#ifndef KALDI_CUDA_DECODER_BATCHED_STATIC_NNET3_KERNELS_H_
#define KALDI_CUDA_DECODER_BATCHED_STATIC_NNET3_KERNELS_H_

namespace kaldi {
namespace cuda_decoder {

// Describe what each batch slot is made of. Used by the context switch kernels
struct BatchSlotAssignment {
  BaseFloat *d_features;
  BaseFloat *d_ivectors;
  int ichannel;
  int n_frames_already_in_context;
  int n_new_frames;
};

struct BatchedStaticNnet3KernelParams {
  const BaseFloat *d_all_new_features;
  const BatchSlotAssignment *d_batch_slot_assignement;
  BaseFloat *d_all_context_frames;
  BaseFloat *d_batch_with_context;
  BaseFloat *d_batch_ivectors;
  int d_batch_ivectors_stride;
  int batch_size;
  int d_features_frame_stride;
  int d_ivectors_frame_stride;
  int d_all_context_frames_frame_stride;
  int d_batch_with_context_frame_stride;
  int d_all_context_frames_channel_stride;
  int d_batch_with_context_batch_stride;
  int input_dim;
  int ivector_dim;
  int total_nnet_context;
  int total_nnet_left_context;
  int total_nnet_right_context;
  int input_frames_per_chunk_with_context;
};

// Takes as a input strided new chunks ptrs [chk0, chk1, chk2..]
// associated to channels [ch0, ch1, ch2...]
// And build a continuous batch such as:
// Batch with context:
// row0: [left_context(ch0), chk0]
// row0: [left_context(ch1), chk1]
// row0: [left_context(ch2), chk2]
// With left context being either part of a previous chunk for that channel, or
// just duplications of frame0 if this is the first chunk for that channel The
// end of each chunk for each row will then be used as a right context
void BuildBatchWithContextKernel(const dim3 &grid, const dim3 &block,
                                 const cudaStream_t &stream,
                                 const BatchedStaticNnet3KernelParams &params);

// Same thing than BuildBatchWithContextKernelContextFlush, except that the
// final frame is replicated to create the right context
void BuildBatchWithContextKernelContextFlush(
    const dim3 &grid, const dim3 &block, const cudaStream_t &stream,
    const BatchedStaticNnet3KernelParams &params);
void SaveContextFromBatchKernel(const dim3 &grid, const dim3 &block,
                                const cudaStream_t &stream,
                                const BatchedStaticNnet3KernelParams &params);

}  // namespace cuda_decoder
}  // namespace kaldi

#endif  // KALDI_CUDA_DECODER_BATCHED_STATIC_NNET3_KERNELS_H_
#endif  // HAVE_CUDA
