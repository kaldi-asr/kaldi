// cudadecoder/cuda-decoder-kernels.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun, Justin Luitjens, Ryan Leary
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

#ifndef KALDI_CUDA_DECODER_CUDA_DECODER_KERNELS_H_
#define KALDI_CUDA_DECODER_CUDA_DECODER_KERNELS_H_

#include "cuda-decoder.h"

namespace kaldi {
namespace CudaDecode {
__global__ void get_best_cost_kernel(DeviceParams cst_dev_params,
                                     KernelParams params, bool isfinal,
                                     CostType fst_zero);
__global__ void finalize_process_non_emitting_kernel(
    DeviceParams cst_dev_params, KernelParams params);
__global__ void save_channels_state_from_lanes_kernel(
    DeviceParams cst_dev_params, KernelParams params);
__global__ void load_channels_state_in_lanes_kernel(DeviceParams cst_dev_params,
                                                    KernelParams params);
__global__ void init_decoding_on_device_kernel(DeviceParams cst_dev_params,
                                               KernelParams params);
__global__ void initialize_initial_lane_kernel(DeviceParams cst_dev_params);
template <bool IS_EMITTING>
__global__ void expand_arcs_kernel(DeviceParams cst_dev_params,
                                   KernelParams params);
template <bool IS_EMITTING>
__global__ void post_expand_kernel(DeviceParams cst_dev_params,
                                   KernelParams params);
__global__ void fill_hashmap_with_main_q_kernel(DeviceParams cst_dev_params,
                                                KernelParams params);
__global__ void emitting_preprocess_and_list_extra_prev_tokens_step1_kernel(
    DeviceParams cst_dev_params, KernelParams params);
__global__ void emitting_preprocess_and_list_extra_prev_tokens_step2_kernel(
    DeviceParams cst_dev_params, KernelParams params);
__global__ void emitting_preprocess_and_list_extra_prev_tokens_step3_kernel(
    DeviceParams cst_dev_params, KernelParams params);
__global__ void emitting_preprocess_and_list_extra_prev_tokens_step4_kernel(
    DeviceParams cst_dev_params, KernelParams params);
__global__ void nonemitting_preprocess_and_contract_kernel(
    DeviceParams cst_dev_params, KernelParams params);
template <typename T>
__global__ void concatenate_lanes_data(DeviceParams cst_dev_params,
                                       KernelParams params,
                                       LaneMatrixView<T> src, T *concat);

__global__ void init_hashmap_kernel(DeviceParams cst_dev_params);
__global__ void clear_hashmap_kernel(DeviceParams cst_dev_params,
                                     KernelParams params);
__global__ void compute_costs_histogram_kernel(DeviceParams cst_dev_params,
                                               KernelParams params,
                                               bool use_aux_q);
__global__ void update_beam_using_histogram_kernel(DeviceParams cst_dev_params,
                                                   KernelParams params,
                                                   bool use_aux_q);

__global__ void get_best_cost_kernel_step1(DeviceParams cst_dev_params,
                                           KernelParams params,
                                           bool use_final_probs,
                                           CostType fst_zero);
__global__ void get_best_cost_kernel_step2(DeviceParams cst_dev_params,
                                           KernelParams params,
                                           bool use_final_probs,
                                           CostType fst_zero);

struct DeviceParams {
  ChannelMatrixView<ChannelCounters> d_channels_counters;
  LaneMatrixView<LaneCounters> d_lanes_counters;

  ChannelMatrixView<int2> d_main_q_state_and_cost;
  ChannelMatrixView<int32> d_main_q_degrees_prefix_sum;
  ChannelMatrixView<int32> d_main_q_arc_offsets;
  LaneMatrixView<CostType> d_main_q_acoustic_cost;
  LaneMatrixView<InfoToken> d_main_q_info;
  LaneMatrixView<int2> d_aux_q_state_and_cost;
  LaneMatrixView<CostType> d_aux_q_acoustic_cost;
  LaneMatrixView<InfoToken> d_aux_q_info;
  LaneMatrixView<HashmapValueT> d_hashmap_values;
  LaneMatrixView<int2> d_list_final_tokens_in_main_q;
  LaneMatrixView<float2> d_main_q_extra_and_acoustic_cost;
  LaneMatrixView<int32> d_histograms;
  LaneMatrixView<int2> d_main_q_block_sums_prefix_sum;
  LaneMatrixView<int32> d_main_q_state_hash_idx;
  LaneMatrixView<int32> d_main_q_extra_prev_tokens_prefix_sum;
  LaneMatrixView<int32> d_main_q_n_extra_prev_tokens_local_idx;
  LaneMatrixView<InfoToken> d_main_q_extra_prev_tokens;

  int32 max_nlanes;
  int32 q_capacity;
  CostType *d_arc_weights;
  int32 *d_arc_nextstates;
  int32 *d_arc_pdf_ilabels;
  uint32 *d_arc_e_offsets;
  uint32 *d_arc_ne_offsets;
  CostType *d_fst_final_costs;
  int32 nstates;
  CostType default_beam;
  CostType lattice_beam;
  int32 init_channel_id;
  StateId init_state;
  CostType init_cost;
  int32 hashmap_capacity;
  int32 max_active;
  int32 adaptive_beam_static_segment;
  int32 adaptive_beam_bin_width;
};

struct KernelParams {
  // In AdvanceDecoding,
  // the lane lane_id will compute the channel
  // with channel_id = channel_to_compute[lane_id]
  ChannelId channel_to_compute[KALDI_CUDA_DECODER_MAX_N_LANES];
  int32 main_q_end_lane_offsets[KALDI_CUDA_DECODER_MAX_N_LANES];
  BaseFloat *loglikelihoods_ptrs[KALDI_CUDA_DECODER_MAX_N_LANES];
  int32 nlanes_used;
};

typedef unsigned char BinId;
}  // namespace kaldi
}  // namespace CudaDecode

#endif  // KALDI_CUDA_DECODER_CUDA_DECODER_KERNELS_H_
