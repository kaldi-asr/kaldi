// cudadecoder/batched-static-nnet3-kernels.cu
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cudadecoder/batched-static-nnet3-kernels.h"

#include <stdio.h>
namespace kaldi {
namespace cuda_decoder {

__global__ void build_batch_with_context_kernel(
    BatchedStaticNnet3KernelParams params) {
  for (int batch_slot = blockIdx.z; batch_slot < params.batch_size;
       batch_slot += gridDim.z) {
    BatchSlotAssignment batch_assign =
        params.d_batch_slot_assignement[batch_slot];
    const BaseFloat *d_batch_slot_features = batch_assign.d_features;
    BaseFloat *d_channel_context =
        &params
             .d_all_context_frames[batch_assign.ichannel *
                                   params.d_all_context_frames_channel_stride];
    BaseFloat *d_batch_slot_with_context =
        &params.d_batch_with_context[params.d_batch_with_context_batch_stride *
                                     batch_slot];

    int n_frames_available =
        batch_assign.n_frames_already_in_context + batch_assign.n_new_frames;
    int n_frames_to_set = n_frames_available;
    int n_left_context_frames_from_frame0 = 0;
    if (batch_assign.n_frames_already_in_context == 0) {
      // First chunk for that utterance. Generating left context by duplicating
      // frame0
      n_frames_to_set += params.total_nnet_left_context;
      n_left_context_frames_from_frame0 = params.total_nnet_left_context;
    }

    for (int iframe = blockIdx.y; iframe < n_frames_to_set;
         iframe += gridDim.y) {
      for (int idim = threadIdx.x; idim < params.input_dim;
           idim += blockDim.x) {
        if (iframe < n_left_context_frames_from_frame0) {
          d_batch_slot_with_context
              [iframe * params.d_batch_with_context_frame_stride + idim] =
                  d_batch_slot_features[0 + idim];  // frame 0
        } else if (iframe < (n_left_context_frames_from_frame0 +
                             batch_assign.n_frames_already_in_context)) {
          // Those are the frames coming from context
          int src_iframe_in_saved_context =
              iframe - n_left_context_frames_from_frame0;
          d_batch_slot_with_context[iframe *
                                        params
                                            .d_batch_with_context_frame_stride +
                                    idim] =
              d_channel_context[src_iframe_in_saved_context *
                                    params.d_all_context_frames_frame_stride +
                                idim];
        } else {
          // Now we are moving the frames coming from the new chunk
          int src_iframe_in_new_chunk =
              iframe - n_left_context_frames_from_frame0 -
              batch_assign.n_frames_already_in_context;
          d_batch_slot_with_context
              [iframe * params.d_batch_with_context_frame_stride + idim] =
                  d_batch_slot_features[src_iframe_in_new_chunk *
                                            params.d_features_frame_stride +
                                        idim];
        }
      }

      if (iframe == 0 &&
          params.d_batch_ivectors) {  // one CTA moves the ivectors
        for (int idim = threadIdx.x; idim < params.ivector_dim;
             idim += blockDim.x) {
          params.d_batch_ivectors[batch_slot * params.d_batch_ivectors_stride +
                                  idim] = batch_assign.d_ivectors[idim];
        }
      }
    }
  }
}

void BuildBatchWithContextKernel(const dim3 &grid, const dim3 &block,
                                 const cudaStream_t &stream,
                                 const BatchedStaticNnet3KernelParams &params) {
  build_batch_with_context_kernel<<<grid, block, 0, stream>>>(params);
}

__global__ void build_batch_with_context_context_flush_kernel(
    BatchedStaticNnet3KernelParams params) {
  for (int batch_slot = blockIdx.z; batch_slot < params.batch_size;
       batch_slot += gridDim.z) {
    BatchSlotAssignment batch_assign =
        params.d_batch_slot_assignement[batch_slot];
    BaseFloat *d_channel_context =
        &params
             .d_all_context_frames[batch_assign.ichannel *
                                   params.d_all_context_frames_channel_stride];
    BaseFloat *d_batch_slot_with_context =
        &params.d_batch_with_context[params.d_batch_with_context_batch_stride *
                                     batch_slot];

    int n_frames_in_context = batch_assign.n_frames_already_in_context;
    int n_frames_to_set = n_frames_in_context + params.total_nnet_right_context;

    for (int iframe = blockIdx.y; iframe < n_frames_to_set;
         iframe += gridDim.y) {
      for (int idim = threadIdx.x; idim < params.input_dim;
           idim += blockDim.x) {
        if (iframe < n_frames_in_context) {
          d_batch_slot_with_context
              [iframe * params.d_batch_with_context_frame_stride +
               idim] = d_channel_context
                  [iframe * params.d_all_context_frames_frame_stride + idim];
        } else if (iframe < n_frames_to_set) {
          // Generating right context from last frame
          int src_iframe_in_saved_context = n_frames_in_context - 1;
          d_batch_slot_with_context[iframe *
                                        params
                                            .d_batch_with_context_frame_stride +
                                    idim] =
              d_channel_context[src_iframe_in_saved_context *
                                    params.d_all_context_frames_frame_stride +
                                idim];
        }
      }

      if (iframe == 0 &&
          params.d_batch_ivectors) {  // one CTA moves the ivectors
        for (int idim = threadIdx.x; idim < params.ivector_dim;
             idim += blockDim.x) {
          params.d_batch_ivectors[batch_slot * params.d_batch_ivectors_stride +
                                  idim] = batch_assign.d_ivectors[idim];
        }
      }
    }
  }
}

void BuildBatchWithContextKernelContextFlush(
    const dim3 &grid, const dim3 &block, const cudaStream_t &stream,
    const BatchedStaticNnet3KernelParams &params) {
  build_batch_with_context_context_flush_kernel<<<grid, block, 0, stream>>>(
      params);
}

__global__ void save_context_from_batch_kernel(
    BatchedStaticNnet3KernelParams params) {
  for (int batch_slot = blockIdx.z; batch_slot < params.batch_size;
       batch_slot += gridDim.z) {
    BatchSlotAssignment batch_assign =
        params.d_batch_slot_assignement[batch_slot];

    // Real frames : does not include frame0 copies for left context
    int n_real_frames_available =
        batch_assign.n_frames_already_in_context + batch_assign.n_new_frames;
    // total frames : includes frame0 copies
    int total_frames_in_batch_slot = n_real_frames_available;
    if (batch_assign.n_frames_already_in_context == 0) {
      // First chunk for that utterance. We generated left context by
      // duplicating frame0
      total_frames_in_batch_slot += params.total_nnet_left_context;
    }
    // total frames : includes frame0 copies
    int n_to_copy = min(total_frames_in_batch_slot, params.total_nnet_context);
    int copy_from_frame = total_frames_in_batch_slot - n_to_copy;
    BaseFloat *d_batch_slot_with_context =
        &params.d_batch_with_context[params.d_batch_with_context_batch_stride *
                                     batch_slot];
    BaseFloat *d_channel_context =
        &params
             .d_all_context_frames[batch_assign.ichannel *
                                   params.d_all_context_frames_channel_stride];

    for (int dst_iframe = blockIdx.y; dst_iframe < n_to_copy;
         dst_iframe += gridDim.y) {
      int src_iframe = copy_from_frame + dst_iframe;
      for (int idim = threadIdx.x; idim < params.input_dim;
           idim += blockDim.x) {
        d_channel_context[dst_iframe *
                              params.d_all_context_frames_frame_stride +
                          idim] = d_batch_slot_with_context
            [src_iframe * params.d_batch_with_context_frame_stride + idim];
      }
    }
  }
}

void SaveContextFromBatchKernel(const dim3 &grid, const dim3 &block,
                                const cudaStream_t &stream,
                                const BatchedStaticNnet3KernelParams &params) {
  save_context_from_batch_kernel<<<grid, block, 0, stream>>>(params);
}

}  // namespace cuda_decoder
}  // namespace kaldi
