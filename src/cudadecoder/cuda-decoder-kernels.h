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

#include "cudadecoder/cuda-decoder-common.h"
#include "util/stl-utils.h"

namespace kaldi {
namespace cuda_decoder {

// DeviceParams contains all top-level const data used by the kernels
// i.e. the data that won't change between kernel calls (such as memory pointers
// to the main_q)
struct DeviceParams {
  ChannelMatrixView<ChannelCounters> d_channels_counters;
  LaneMatrixView<LaneCounters> d_lanes_counters;
  LaneMatrixView<LaneCounters> h_lanes_counters;

  ChannelMatrixView<int2> d_main_q_state_and_cost;
  ChannelMatrixView<int32> d_main_q_degrees_prefix_sum;
  ChannelMatrixView<int32> d_main_q_arc_offsets;
  LaneMatrixView<CostType> d_main_q_acoustic_cost;
  LaneMatrixView<InfoToken> d_main_q_info;
  LaneMatrixView<int2> d_aux_q_state_and_cost;
  LaneMatrixView<InfoToken> d_aux_q_info;
  LaneMatrixView<HashmapValueT> d_hashmap_values;
  LaneMatrixView<int2> h_list_final_tokens_in_main_q;
  LaneMatrixView<float2> d_main_q_extra_and_acoustic_cost;
  LaneMatrixView<int32> d_histograms;
  LaneMatrixView<int2> d_main_q_block_sums_prefix_sum;
  LaneMatrixView<int32> d_main_q_state_hash_idx;
  LaneMatrixView<int32> d_main_q_extra_prev_tokens_prefix_sum;
  LaneMatrixView<int32> d_main_q_n_extra_prev_tokens_local_idx;
  LaneMatrixView<InfoToken> d_main_q_extra_prev_tokens;

  int32 max_nlanes;
  int32 main_q_capacity, aux_q_capacity;
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

  CostType fst_zero;
};

// KernelParams contains all the kernels arguments that change between kernel
// calls
struct KernelParams {
  int32 nlanes_used;
};

// Kernel wrappers
void SaveChannelsStateFromLanesKernel(const dim3 &grid, const dim3 &block,
                                      const cudaStream_t &st,
                                      const DeviceParams &cst_dev_params,
                                      const KernelParams &kernel_params);

void LoadChannelsStateInLanesKernel(const dim3 &grid, const dim3 &block,
                                    const cudaStream_t &st,
                                    const DeviceParams &cst_dev_params,
                                    const KernelParams &kernel_params);

void InitDecodingOnDeviceKernel(const dim3 &grid, const dim3 &block,
                                const cudaStream_t &st,
                                const DeviceParams &cst_dev_params,
                                const KernelParams &kernel_params);

void InitializeInitialLaneKernel(const dim3 &grid, const dim3 &block,
                                 const cudaStream_t &st,
                                 const DeviceParams &cst_dev_params);

void ResetForFrameAndEstimateCutoffKernel(const dim3 &grid, const dim3 &block,
                                          const cudaStream_t &st,
                                          const DeviceParams &cst_dev_params,
                                          const KernelParams &kernel_params);

template <bool IS_EMITTING>
void ExpandArcsKernel(const dim3 &grid, const dim3 &block,
                      const cudaStream_t &st,
                      const DeviceParams &cst_dev_params,
                      const KernelParams &kernel_params);

template <bool IS_EMITTING>
void PostExpandKernel(const dim3 &grid, const dim3 &block,
                      const cudaStream_t &st,
                      const DeviceParams &cst_dev_params,
                      const KernelParams &kernel_params);

void PostContractAndPreprocessKernel(const dim3 &grid, const dim3 &block,
                                     const cudaStream_t &st,
                                     const DeviceParams &cst_dev_params,
                                     const KernelParams &kernel_params);

void NonEmittingPreprocessAndContractKernel(const dim3 &grid, const dim3 &block,
                                            const cudaStream_t &st,
                                            const DeviceParams &cst_dev_params,
                                            const KernelParams &kernel_params);

void FillHashmapWithMainQKernel(const dim3 &grid, const dim3 &block,
                                const cudaStream_t &st,
                                const DeviceParams &cst_dev_params,
                                const KernelParams &kernel_params);

void EmittingPreprocessAndListExtraPrevTokensStep1Kernel(
    const dim3 &grid, const dim3 &block, const cudaStream_t &st,
    const DeviceParams &cst_dev_params, const KernelParams &kernel_params);

void EmittingPreprocessAndListExtraPrevTokensStep2Kernel(
    const dim3 &grid, const dim3 &block, const cudaStream_t &st,
    const DeviceParams &cst_dev_params, const KernelParams &kernel_params);

void EmittingPreprocessAndListExtraPrevTokensStep3Kernel(
    const dim3 &grid, const dim3 &block, const cudaStream_t &st,
    const DeviceParams &cst_dev_params, const KernelParams &kernel_params);

void EmittingPreprocessAndListExtraPrevTokensStep4Kernel(
    const dim3 &grid, const dim3 &block, const cudaStream_t &st,
    const DeviceParams &cst_dev_params, const KernelParams &kernel_params);

void ComputeLaneOffsetsKernel(const dim3 &grid, const dim3 &block,
                              const cudaStream_t &st,
                              const DeviceParams &cst_dev_params,
                              const KernelParams &kernel_params);

template <typename T>
void ConcatenateLanesDataKernel(const dim3 &grid, const dim3 &block,
                                const cudaStream_t &st,
                                const DeviceParams &cst_dev_params,
                                const KernelParams &kernel_params,
                                const LaneMatrixView<T> &src, T *concat,
                                int32 *lane_offsets);

void InitHashmapKernel(const dim3 &grid, const dim3 &block,
                       const cudaStream_t &st,
                       const DeviceParams &cst_dev_params);

void ClearHashmapKernel(const dim3 &grid, const dim3 &block,
                        const cudaStream_t &st,
                        const DeviceParams &cst_dev_params,
                        const KernelParams &kernel_params);

void ComputeCostsHistogramKernel(const dim3 &grid, const dim3 &block,
                                 const cudaStream_t &st,
                                 const DeviceParams &cst_dev_params,
                                 const KernelParams &kernel_params,
                                 bool use_aux_q);

void UpdateBeamUsingHistogramKernel(const dim3 &grid, const dim3 &block,
                                    const cudaStream_t &st,
                                    const DeviceParams &cst_dev_params,
                                    const KernelParams &kernel_params,
                                    bool use_aux_q);

void FinalizeProcessNonEmittingKernel(const dim3 &grid, const dim3 &block,
                                      const cudaStream_t &st,
                                      const DeviceParams &cst_dev_params,
                                      const KernelParams &kernel_params);

void GetBestCostStep1Kernel(const dim3 &grid, const dim3 &block,
                            const cudaStream_t &st,
                            const DeviceParams &cst_dev_params,
                            const KernelParams &kernel_params, bool isfinal);

void GetBestCostStep2Kernel(const dim3 &grid, const dim3 &block,
                            const cudaStream_t &st,
                            const DeviceParams &cst_dev_params,
                            const KernelParams &kernel_params, bool isfinal);

void GetBestCostStep3Kernel(const dim3 &grid, const dim3 &block,
                            const cudaStream_t &st,
                            const DeviceParams &cst_dev_params,
                            const KernelParams &kernel_params);

typedef unsigned char BinId;

}  // namespace cuda_decoder
}  // namespace kaldi

#endif  // KALDI_CUDA_DECODER_CUDA_DECODER_KERNELS_H_
