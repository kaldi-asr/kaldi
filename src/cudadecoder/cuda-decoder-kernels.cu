// cudadecoder/cuda-decoder-kernels.cu
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

#include <cub/cub.cuh>
#include "cuda-decoder-kernels.h"
#include "cuda-decoder-kernels-utils.h"

namespace kaldi {
namespace cuda_decoder {

// Initialize the hashmap with NO_VAL
// Called in InitDeviceData, when building the CudaDecoder object
__global__ void init_hashmap_kernel(DeviceParams cst_dev_params) {
  const int max_nlanes = cst_dev_params.max_nlanes;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, max_nlanes) {
    const int capacity = cst_dev_params.hashmap_capacity;
    KALDI_CUDA_DECODER_1D_KERNEL_LOOP(idx, capacity) {
      cst_dev_params.d_hashmap_values.lane(ilane)[idx] =
          KALDI_CUDA_DECODER_HASHMAP_NO_VAL;
    }
  }
}

// Initialize initial channel on  device
// Called by ComputeInitialChannel
// It is NOT called in InitDecoding
// In InitDecoding we will clone the initial channel into the channel we called
// InitDecoding on
// Here we are actually creating this initial channel
// we do that once in the CudaDecoder constructor.
//
// The initial channel is the state of a channel when
// it will start decoding a new utterance
// thread (1, 1, 1)
// blocks(1, 1, 1);
__global__ void initialize_initial_lane_kernel(DeviceParams cst_dev_params) {
  const int init_ichannel = cst_dev_params.init_channel_id;
  const int init_ilane = 0;
  ChannelCounters *init_channel_counters =
      cst_dev_params.d_channels_counters.channel(init_ichannel);
  LaneCounters *lane_counters =
      cst_dev_params.d_lanes_counters.lane(init_ilane);

  // Making the data look like an ExpandArcsEmitting just executed,
  // and put the StartState in the aux_q. We will then pick up a normal
  // execution from there
  // (calling PruneAndPreprocess, then ExpandArcsNonEmitting..)
  lane_counters->aux_q_end = 0;
  lane_counters->aux_q_requested = 0;
  lane_counters->post_expand_aux_q_end = 1;
  lane_counters->main_q_global_offset = 0;
  lane_counters->main_q_local_offset = 0;
  lane_counters->main_q_n_extra_prev_tokens = 0;
  lane_counters->int_cutoff = INT_MAX;
  lane_counters->main_q_n_emitting_tokens = 0;  // all non emitting
  lane_counters->int_beam = floatToOrderedInt(cst_dev_params.default_beam);
  lane_counters->main_q_narcs_and_end = {0, 0};
  lane_counters->main_q_requested = 0;
  lane_counters->prev_arg_min_int_cost = 0;
  const StateId init_state = cst_dev_params.init_state;
  const CostType init_cost = cst_dev_params.init_cost;
  IntegerCostType int_init_cost = floatToOrderedInt(init_cost);
  cst_dev_params.d_aux_q_state_and_cost.lane(init_ilane)[0] = {init_state,
                                                               int_init_cost};
  lane_counters->min_int_cost = int_init_cost;
  CostType cutoff = orderedIntToFloat(int_init_cost);
  lane_counters->int_cutoff =
      floatToOrderedInt(cutoff + cst_dev_params.default_beam);
  cst_dev_params.d_aux_q_info.lane(init_ilane)[0] = {INT_MIN, -1};
}

// Called by InitDecoding
// Called when some channels will start decoding a new utterance
// do everything that's needed to do on the device to start decoding a new
// utterance with those channels
// It clones the initial channel (created in initialize_initial_lane_kernel)
// into the channels we want to InitDecoding on
__global__ void init_decoding_on_device_kernel(DeviceParams cst_dev_params,
                                               KernelParams params) {
  const int init_ichannel = cst_dev_params.init_channel_id;

  const ChannelCounters *init_channel_counters =
      cst_dev_params.d_channels_counters.channel(init_ichannel);
  const int32 init_main_q_end =
      init_channel_counters->prev_main_q_narcs_and_end.y;
  const int32 nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    KALDI_CUDA_DECODER_1D_KERNEL_LOOP(idx, init_main_q_end) {
      const LaneCounters *lane_counters =
          cst_dev_params.d_lanes_counters.lane(ilane);
      const int32 ichannel = lane_counters->channel_to_compute;
      cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[idx] =
          cst_dev_params.d_main_q_state_and_cost.channel(init_ichannel)[idx];
      cst_dev_params.d_main_q_degrees_prefix_sum.channel(ichannel)[idx] =
          cst_dev_params.d_main_q_degrees_prefix_sum.channel(
              init_ichannel)[idx];
      cst_dev_params.d_main_q_arc_offsets.channel(ichannel)[idx] =
          cst_dev_params.d_main_q_arc_offsets.channel(init_ichannel)[idx];
      if (idx == 0) {
        ChannelCounters *channel_counters =
            cst_dev_params.d_channels_counters.channel(ichannel);
        channel_counters->prev_main_q_narcs_and_end =
            init_channel_counters->prev_main_q_narcs_and_end;
        channel_counters->prev_main_q_n_extra_prev_tokens =
            init_channel_counters->prev_main_q_n_extra_prev_tokens;
        channel_counters->prev_main_q_global_offset = 0;
        channel_counters->prev_main_q_extra_prev_tokens_global_offset = 0;
        channel_counters->prev_beam = cst_dev_params.default_beam;
      }
    }
  }
}

// Context switch : load
// Called by LoadChannelsStateToLanes
// THREADS : (1, 1, 1)
// BLOCKS : (1, nlanes_used, 1)
__global__ void load_channels_state_in_lanes_kernel(DeviceParams cst_dev_params,
                                                    KernelParams params) {
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    const int32 ichannel = lane_counters->channel_to_compute;
    const ChannelCounters *channel_counters =
        cst_dev_params.d_channels_counters.channel(ichannel);
    int2 main_q_narcs_and_end = channel_counters->prev_main_q_narcs_and_end;
    lane_counters->main_q_narcs_and_end = main_q_narcs_and_end;
    lane_counters->main_q_n_extra_prev_tokens =
        channel_counters->prev_main_q_n_extra_prev_tokens;
    CostType beam = channel_counters->prev_beam;
    IntegerCostType int_beam = floatToOrderedInt(beam);
    lane_counters->int_beam = int_beam;
    lane_counters->adaptive_int_beam_with_validity_index.x = int_beam;
    lane_counters->adaptive_int_beam_with_validity_index.y =
        cst_dev_params.adaptive_beam_static_segment;
    lane_counters->main_q_global_offset =
        channel_counters
            ->prev_main_q_global_offset;  // we'll update it after emitting
    lane_counters->main_q_extra_prev_tokens_global_offset =
        channel_counters->prev_main_q_extra_prev_tokens_global_offset;
  }
}

// Context switch : store
// Called by SaveChannelsStateFromLanes
// THREADS : (1, 1, 1)
// BLOCKS : (1, nchannel_to_compute, 1)
__global__ void save_channels_state_from_lanes_kernel(
    DeviceParams cst_dev_params, KernelParams params) {
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    const LaneCounters *lane_counters =
        cst_dev_params.d_lanes_counters.lane(ilane);
    const int32 ichannel = lane_counters->channel_to_compute;
    ChannelCounters *channel_counters =
        cst_dev_params.d_channels_counters.channel(ichannel);
    channel_counters->prev_main_q_global_offset =
        lane_counters->main_q_global_offset;
    channel_counters->prev_main_q_extra_prev_tokens_global_offset =
        lane_counters->main_q_extra_prev_tokens_global_offset;
    channel_counters->prev_main_q_narcs_and_end =
        lane_counters->main_q_narcs_and_end;
    channel_counters->prev_main_q_n_extra_prev_tokens =
        lane_counters->main_q_n_extra_prev_tokens;
    channel_counters->prev_beam = orderedIntToFloat(lane_counters->int_beam);
  }
}

// compute_lane_offsets_kernel
// the kernel concatenate_lanes_data concatenates multiple array into a single
// continuous array
// compute_lane_offsets_kernel computes the offset of each array into this
// continous array
// This kernel is 1D : the lanes are on the X dimension, because we want to
// compute the offset of those lanes
__global__ void compute_lane_offsets_kernel(DeviceParams cst_dev_params,
                                            KernelParams params) {
  typedef cub::BlockScan<int4, KALDI_CUDA_DECODER_1D_BLOCK> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const int nlanes = params.nlanes_used;
  int4 sum_so_far = {0, 0, 0, 0};
  KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(
      block_offset, thread_idx,
      nlanes + 1) {  // +1 because we are doing an exclusive sum, and we want
                     // all the values
    int32 ilane = block_offset + thread_idx;
    int4 zero4 = {0, 0, 0, 0};
    int4 lane_offsets = zero4;
    if (ilane < nlanes) {  // nlanes, not nlanes+1, because we cannot read +1
                           // values (undefined)
      LaneCounters *d_lane_counters =
          cst_dev_params.d_lanes_counters.lane(ilane);
      int32 main_q_end = d_lane_counters->main_q_narcs_and_end.y;
      int32 n_emitting_tokens = d_lane_counters->main_q_n_emitting_tokens;
      int32 main_q_n_extra_prev_tokens =
          d_lane_counters->main_q_n_extra_prev_tokens;
      lane_offsets = {main_q_end, n_emitting_tokens, main_q_n_extra_prev_tokens,
                      0};
    }
    int4 block_aggregate;
    BlockScan(temp_storage)
        .ExclusiveScan(lane_offsets, lane_offsets, zero4, PlusPlusPlusPlus(),
                       block_aggregate);
    PlusPlusPlusPlus pppp;
    lane_offsets = pppp(lane_offsets, sum_so_far);
    sum_so_far = pppp(sum_so_far, block_aggregate);
    if (ilane < (nlanes + 1)) {  // nlanes+1, to write the output
      LaneCounters *d_lane_counters =
          cst_dev_params.d_lanes_counters.lane(ilane);
      LaneCounters *h_lane_counters =
          cst_dev_params.h_lanes_counters.lane(ilane);
      h_lane_counters->main_q_end_lane_offset =
          d_lane_counters->main_q_end_lane_offset = lane_offsets.x;
      h_lane_counters->main_q_n_emitting_tokens_lane_offset =
          d_lane_counters->main_q_n_emitting_tokens_lane_offset =
              lane_offsets.y;
      h_lane_counters->main_q_n_extra_prev_tokens_lane_offset =
          d_lane_counters->main_q_n_extra_prev_tokens_lane_offset =
              lane_offsets.z;
    }
    __syncthreads();  // reusing temp_storage
  }
}

// concatenate_lanes_data
// Called by PerformConcatenatedCopy
// Creates a concatenate array into concat,
// by concatenating all the arrays src.lane(ilane)
// for ilane=0..params.nlanes_used
// Used to prepare data for copy to Host. We want to avoid small Device2Host
// copies.
template <typename T>
__global__ void concatenate_lanes_data_kernel(DeviceParams cst_dev_params,
                                              KernelParams params,
                                              LaneMatrixView<T> src, T *concat,
                                              int32 *lane_offsets) {
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    const int32 stride =
        sizeof(LaneCounters) / sizeof(int32);  // offsets are in LaneCounters
    int32 beg = *(lane_offsets + ilane * stride);
    int32 end = *(lane_offsets + (ilane + 1) * stride);
    int32 vec_size = end - beg;
    KALDI_CUDA_DECODER_1D_KERNEL_LOOP(idx, vec_size) {
      T d = src.lane(ilane)[idx];
      concat[beg + idx] = d;
    }
  }
}

// nonemitting_preprocess_and_contract_kernel
// Called from PruneAndPreprocess
// This kernels prune the aux_q, move the survival tokens to the main_q,
// and add the preprocessing information necessary for the next ExpandArcs
// (the expand that follows PruneAndPreprocess is always non-emitting)
// It prunes the tokens using the cutoff, and prepare the data necessary for
// ExpandArcs:
// d_main_q_degrees_prefix_sum, d_main_q_arc_offsets_
// The prefix sum is done in one-pass here, using a trick (we compute the prefix
// sum
// as we fill the main_q)
__global__ void nonemitting_preprocess_and_contract_kernel(
    DeviceParams cst_dev_params, KernelParams params) {
  typedef cub::BlockScan<int2, KALDI_CUDA_DECODER_1D_BLOCK> BlockScan;
  __shared__ typename BlockScan::TempStorage sh_temp_storage;
  // We need to move the survival tokens to the main_q
  //
  // sh_main_q_global_block_offset has two purposes :
  // (1) to know where to store the survival tokens in the main_q
  // (2) to perform the prefix sum degrees (of the survival tokens)
  __shared__ int2 sh_main_q_global_block_offset;
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    const int32 aux_q_end = lane_counters->post_expand_aux_q_end;
    const IntegerCostType int_cutoff = lane_counters->int_cutoff;
    // Keeping whole CTA alive. We'll use __syncthreads()
    KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(block_offset, thread_idx,
                                                   aux_q_end) {
      const int32 aux_q_idx = block_offset + thread_idx;
      const int32 ichannel = lane_counters->channel_to_compute;
      int32 degree = 0;
      int32 arc_start = -1;
      StateId token_state;
      IntegerCostType token_int_cost;
      // We've kept the whole CTA alive. Now we keep only those will a valid
      // token
      if (aux_q_idx < aux_q_end) {
        const int2 both =
            cst_dev_params.d_aux_q_state_and_cost.lane(ilane)[aux_q_idx];
        token_state = both.x;
        token_int_cost = both.y;

        if (token_int_cost < int_cutoff) {
          // We'll keep that token. Loading its arc degree/csr offset now.
          arc_start = cst_dev_params.d_arc_ne_offsets[token_state];
          const int32 arc_end =
              cst_dev_params.d_arc_ne_offsets[token_state + 1];
          degree = arc_end - arc_start;
        }
      }

      // If we've set a different arc_start,
      // this thread has a valid unpruned token
      int32 is_pruned = (arc_start == -1);

      // We now know which tokens will be moved to the main_q, the remaining
      // will be pruned
      // we now compute a prefix sum inside the CUDA block to determine the
      // local indexes of the unpruned tokens
      // the first unpruned token will have a index of 0, the second 1, ...
      // We also need to compute the prefix sum of the arc degrees
      // we start by doing a local prefix sum inside the CUDA block
      int2 block_prefix_sum_narcs_and_end = {degree, (is_pruned ? 0 : 1)};
      const int2 zero2 = {0, 0};

      // Computing the prefix sum (exclusive)
      BlockScan(sh_temp_storage)
          .ExclusiveScan(block_prefix_sum_narcs_and_end,
                         block_prefix_sum_narcs_and_end, zero2, PlusPlus());

      if (KALDI_CUDA_DECODER_IS_LAST_1D_THREAD()) {
        // This conditional branch is entered by the last thread
        // Because it is the last, the prefix_sum of that thread contains the
        // sum of all elements

        // We also add the value from this thread - the prefix sum is exclusive
        // For the sum, we want it inclusive
        int2 block_sum = block_prefix_sum_narcs_and_end;
        block_sum.x += degree;
        block_sum.y += is_pruned ? 0 : 1;

        // Doing two things at the same time :
        // requesting a spot in the main_q to store the survival tokens from
        // this CTA
        // We also increment the narcs value. atomic64.x will contain the number
        // of
        // arcs in the main_q up until the atomic64.y index
        // That's all we need to finish our prefix sum. We add this global
        // offset.

        // First atomic to check if we are not overflowing main_q.
        int block_offset =
            atomicAdd(&lane_counters->main_q_requested, block_sum.y);

        // Verify that we do not overflow
        if (block_offset + block_sum.y < cst_dev_params.main_q_capacity) {
          // we don't overflow we can safely grab a spot in the main_q
          sh_main_q_global_block_offset =
              atomicAddI2(&lane_counters->main_q_narcs_and_end, block_sum);
        } else {
          // our update would overflow
          lane_counters->q_overflow |= OVERFLOW_MAIN_Q;  // for the host
          sh_main_q_global_block_offset.y =
              cst_dev_params.main_q_capacity;  // used as flag to broadcast the
                                               // information in the CTA
        }
      }

      // Syncing because :
      // - Broadcasting sh_main_q_global_block_offset
      // - We may reuse sh_temp_storage (cf CUB doc)
      __syncthreads();

      // Checking if we are overflowing the main_q
      // All threads are executing the next line
      if (sh_main_q_global_block_offset.y == cst_dev_params.main_q_capacity)
        goto end_lane;  // done for this lane

      // If we are executing the following lines it means that we are not
      // overflowing the queue
      // We then continue what we were doing
      if (!is_pruned) {
        bool moving_emitting_tokens = (lane_counters->main_q_local_offset == 0);
        // we will move our unpruned token to the main_q, at index main_q_idx
        InfoToken tok_info = cst_dev_params.d_aux_q_info.lane(ilane)[aux_q_idx];
        const int32 main_q_idx =
            sh_main_q_global_block_offset.y + block_prefix_sum_narcs_and_end.y;
        CostType acoustic_cost = 0.0f;
        if (moving_emitting_tokens && tok_info.arc_idx != -1) {
          const int32 arc_ilabel =
              cst_dev_params.d_arc_pdf_ilabels[tok_info.arc_idx];
          acoustic_cost = -lane_counters->loglikelihoods[arc_ilabel];
        }
        cst_dev_params.d_main_q_info.lane(ilane)[main_q_idx] = tok_info;

        // Moving the token to the main q
        cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx] = {
            token_state, token_int_cost};
        cst_dev_params.d_main_q_acoustic_cost.lane(ilane)[main_q_idx] =
            acoustic_cost;
        // Saving the global prefix sum
        const int32 prefix_sum_narcs =
            sh_main_q_global_block_offset.x + block_prefix_sum_narcs_and_end.x;
        cst_dev_params.d_main_q_degrees_prefix_sum.channel(
            ichannel)[main_q_idx] = prefix_sum_narcs;
        // Saving the CSR arc offset for that token's state
        // it will be used by the expand kernel, and avoid doing a new random
        // memory access in the expand kernel
        cst_dev_params.d_main_q_arc_offsets.channel(ichannel)[main_q_idx] =
            arc_start;
      }
    }

  end_lane:;  // empty statement
  }
}

// GetAdaptiveBeam is used in ExpandArcs
// When we generate new tokens by traversing arcs, 
// we can end up creating a lot of tokens, if the current frame 
// generated loglikelihoods too uniform for instance (we don't have
// any good tokens that will reduce the cutoff, so we end up generating
// a lot of tokens)
// To avoid overflowing the aux_q, we apply a decreasing beam.
// With aux_q_end being the current aux_q size, we have a decrease function f, with
// adaptive_beam = f(aux_q_end)
// f is a decreasing piecewise constant function
// Please note that when processing tokens, we usually have dozens of thousands of threads
// generating tokens. Those are already in flight, and will not reload the beam immediatly.
// It means that we need to start reducing the beam as soon as we detect that we are generating more tokens than
// expected. 
// We can configure the function f using KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT
// and KALDI_CUDA_DECODER_ADAPTIVE_BEAM_NSTEPS.
// We will use default_beam for the first max_tokens_per_frame/KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT
// tokens in the aux_q.
// Once we reach that number, we will decrease the adaptive beam linearly from default_beam to 0,
// using KALDI_CUDA_DECODER_ADAPTIVE_BEAM_NSTEPS steps
//
// x-axis : aux_q_end. How much tokens are already in the aux_q
// y-axis : adaptive_beam = f(aux_q_end)
// default_beam _| ________________
//               |               /\ _________
//               |                |          _________
//            0 _|   static_segment                   _________
//               |________________________________________________
//               |                                             |     
//   aux_q_end=  0                                    max_tokens_per_frame
// We have :     
// static_segment = max_tokens_per_frame/KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT
// and KALDI_CUDA_DECODER_ADAPTIVE_BEAM_NSTEPS = 3
__device__ void UpdateAdaptiveBeam(const DeviceParams &cst_dev_params,
                                   const int aux_q_index_block_offset,
                                   IntegerCostType min_int_cost,
                                   int2 *adaptive_int_beam_with_validity_index,
                                   LaneCounters *lane_counters) {
  int32 beam_valid_until_idx = adaptive_int_beam_with_validity_index->y;
  if (aux_q_index_block_offset < beam_valid_until_idx) return;  // nothing to do

  CostType beam = orderedIntToFloat(adaptive_int_beam_with_validity_index->x);
  while (aux_q_index_block_offset >= beam_valid_until_idx) {
    beam /= 2;
    beam_valid_until_idx += cst_dev_params.adaptive_beam_bin_width;
  }

  IntegerCostType new_int_cutoff = (min_int_cost < INT_MAX)
      ? floatToOrderedInt(orderedIntToFloat(min_int_cost) + beam)
      : INT_MAX;
  IntegerCostType int_beam = floatToOrderedInt(beam);
  adaptive_int_beam_with_validity_index->x = int_beam;
  adaptive_int_beam_with_validity_index->y = beam_valid_until_idx;
  // We can have races between the two atomics
  // However the worst than can happen is a CTA might delay updating the beam
  // This is not a critical bug. However, once we have a floatToOrderedInt
  // that is generating unsigned ints, we could merge the two atomics into a
  // single atomic64
  atomicMin(&lane_counters->adaptive_int_beam_with_validity_index.x, int_beam);
  atomicMax(&lane_counters->adaptive_int_beam_with_validity_index.y,
            beam_valid_until_idx);
  atomicMin(&lane_counters->int_cutoff, new_int_cutoff);
}

// One CTA / lane
__global__ void reset_for_frame_and_estimate_cutoff_kernel(
    DeviceParams cst_dev_params, KernelParams params) {
  typedef cub::BlockReduce<CostType, KALDI_CUDA_DECODER_1D_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    const int32 ichannel = lane_counters->channel_to_compute;
    ChannelCounters *channel_counters =
        cst_dev_params.d_channels_counters.channel(ichannel);
    if (threadIdx.x == 0) {
      const CostType current_beam = orderedIntToFloat(lane_counters->int_beam);
      // Do some initialization
      lane_counters->q_overflow = OVERFLOW_NONE;
      lane_counters->main_q_n_emitting_tokens = INT_MAX;
      lane_counters->int_cutoff = INT_MAX;
      lane_counters->min_int_cost = INT_MAX;
      lane_counters->q_overflow = OVERFLOW_NONE;
      lane_counters->aux_q_requested = 0;
      lane_counters->main_q_requested = 0;
      lane_counters->main_q_local_offset = 0;
      lane_counters->compute_max_active =
          false;  // will be set to true if necessary
      channel_counters->min_int_cost_and_arg_with_final.x =
          INT_MAX;  // it will be set with atomicMins
      const CostType new_beam =
          fmin(cst_dev_params.default_beam,
               current_beam * KALDI_CUDA_DECODER_ADAPTIVE_BEAM_RECOVER_RATE);
      lane_counters->int_beam = floatToOrderedInt(new_beam);
    }
    const int32 prev_arg_min = lane_counters->prev_arg_min_int_cost;
    int2 both =
        cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[prev_arg_min];
    int32 int_cost = both.y;
    CostType previous_cost = orderedIntToFloat(int_cost);
    const int32 prev_arg_min_state = both.x;
    int32 arc_start = cst_dev_params.d_arc_e_offsets[prev_arg_min_state];
    int32 arc_end = cst_dev_params.d_arc_e_offsets[prev_arg_min_state + 1];
    int32 narcs = arc_end - arc_start;
    // no loop - we only process the first KALDI_CUDA_DECODER_1D_BLOCK arcs
    // we just want an estimate
    CostType total_cost = FLT_MAX;
    if (threadIdx.x < narcs) {
      int32 iarc = arc_start + threadIdx.x;
      CostType arc_fixed_cost = cst_dev_params.d_arc_weights[iarc];
      const int32 arc_ilabel = cst_dev_params.d_arc_pdf_ilabels[iarc];
      CostType acoustic_cost = -lane_counters->loglikelihoods[arc_ilabel];
      total_cost = previous_cost + arc_fixed_cost +
                   acoustic_cost;  // +0.0f, best prev cost is normalized to 0
    }

    KALDI_CUDA_DECODER_1D_KERNEL_LOOP(bin_id, KALDI_CUDA_DECODER_HISTO_NBINS) { 
      cst_dev_params.d_histograms.lane(ilane)[bin_id] = 0; // reset for this frame
    }

    CostType min = BlockReduce(temp_storage).Reduce(total_cost, cub::Min());
    if (narcs > 0 && threadIdx.x == 0) {
      // narcs > 0 to have at least one valid element in the reduce
      CostType new_cutoff = min + orderedIntToFloat(lane_counters->int_beam);
      IntegerCostType new_int_cutoff = floatToOrderedInt(new_cutoff);
      lane_counters->int_cutoff = new_int_cutoff;
      lane_counters->min_int_cost = floatToOrderedInt(min);
    }
  }
}
// ExpandArc kernel
// This kernel does the actual work of traversing arcs
//
// Pseudo code :
// for all token tok in main_q[main_q_offset...end]:
//      u = tok.next_state
//      for all arc a(u->v) in the FST:
//          v_cost = tok.cost + a.cost + accoustic_cost
//
//          if v_cost < cutoff and v_cost < best_state_cost[v]
//              generate token associated to v, add to aux_q
//              if necessary update cutoff
//              if aux_q is getting full, reduce beam
//
// For more information please refer to http://kaldi-asr.org/doc/decoders.html
//
// ExpandArc rely on some preprocessed data to be able to function
// for instance, it needs the prefix sum of the arc degree of all token.state in
// the main_q
// We need to call a Preprocess kernel before ExpandArc
//
// ExpandArc is used for both emitting and nonemitting phases
// Differences between emitting and nonemitting :
//      1) params.d_q_arc_offset contains offsets to either emitting or
//      nonemitting arcs.
//         It is transparent for this kernel. The differentiation was done in
//         the Preprocess kernel,
//         which is responsible for filling the params.d_q_arc_offset array
//      2) Computation of the acoustic cost. If nonemitting, it is equal to 0.
//      If emitting, we need
//         to use values from the acoustic model (through the d_loglikelihoods
//         array)
//
// Note : ExpandArc is not the only kernel able to traverse arcs.
// FinalizeProcessNonemitting contains a simplified version of expand for only
// one CUDA block
template <bool IS_EMITTING>
__global__ void expand_arcs_kernel(DeviceParams cst_dev_params,
                                   KernelParams params) {
  // BlockScan that we will use to compute token indexes in the output queue,
  // and to find the min cost in the block
  typedef cub::BlockScan<int2, KALDI_CUDA_DECODER_1D_BLOCK> BlockScan;
  __shared__ typename BlockScan::TempStorage sh_temp_storage_scan;

  // This kernel writes the new token to the output queue aux_q
  // We will request a spot to store all the new tokens created by threads in
  // this CUDA block
  // sh_aux_q_index_block_offset indicates where to store them in the aux_q
  // tokens created in this CUDA block will be store in :
  // aux_q[sh_aux_q_index_block_offset], aux_q[sh_aux_q_index_block_offset + 1],
  __shared__ int32 sh_aux_q_index_block_offset;
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    const int32 main_q_offset = lane_counters->main_q_local_offset;
    const int32 main_q_end = lane_counters->main_q_narcs_and_end.y;
    const int32 total_narcs = lane_counters->main_q_narcs_and_end.x;
    KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(block_offset, thread_idx,
                                                   total_narcs) {
      int2 adaptive_int_beam_with_validity_index =
          lane_counters->adaptive_int_beam_with_validity_index;
      const int32 ichannel = lane_counters->channel_to_compute;
      // Important : this thread is not responsible for a token in the input
      // queue main_q
      // but for an arc, going out of a token in the main_q
      // The main_q contains in total total_narcs
      // and this thread will compute the main_q_arc_index-th arc of the main_q
      // For instance, first thread in the grid with threadIdx.x == 0 and
      // blockIdx.x == 0
      // will process the first arc of the token in main_q[main_q_offset + 0]
      // (if that token has at least one arc)
      //
      // This insure a perfect one thread = one arc load balancing
      // but we have work to do to know exactly which arc is the
      // main_q_arc_index-th arc
      // (what's its source ? its destination ? its arc_idx the FST CSR ?)
      int32 main_q_arc_index = block_offset + thread_idx;
      // We'll need those variables later in the kernel
      // we declare them outside of the "valid_input" scope
      // to be able to access them later
      int32 main_q_idx;
      int32 arc_idx;
      StateId arc_next_state;
      IntegerCostType int_total_cost = INT_MAX;
      if (main_q_arc_index < total_narcs) {
        // Current thread must take care of main_q_arc_index-th arc
        // we need to now what's the source of that arc
        // ie which token.state in main_q does it start from ?
        // We use a binary search in the prefix sum of the token's degree to get
        // that information
        //
        // Example : main_q contains 3 tokens
        // - First token is associated to a state which has 3 outgoing arc
        // - Second token is associated to a state which has 0 outgoing arc
        // - Third token is associated to a state which has 2 outgoing arc
        //
        // We store the degrees in an array :
        // [3, 0, 2]
        //
        // We then compute the exclusive prefix sum of that array :
        // [0, 3, 3, 5]
        //
        // In total, we have 5 arcs in the main_q. ExpandArc will use 5 threads.
        //
        // Let's say we are the fifth thread in ExpandArc.
        // we have threadIdx.x == 4, and blockIdx.x == 0
        // it gives us main_q_arc_index == 4
        // From there we have no idea what we're supposed to do next, we need to
        // have information about the
        // arc that we're supposed to traverse
        //
        // To do that, we look for the maximum index maxle_i in the prefix sum
        // array such prefix_sum[i] <= 4
        //
        // [0, 3, 3, 5]
        //          |
        //         here
        // maxle_i = 2
        // it means that our source token is at index 2 in the main_q
        // and we are computing the arc at index (main_q_arc_index -
        // prefix_sum[maxle_i]) of that token
        // ie the arc at index (4-3) = 1, the second arc of the second token in
        // main_q

        // Searching for the source of the arc that we will process
        // (main_q_arc_index)
        // we could preprocess the search in the preprocess kernels - for now
        // this kernel is fast enough
        const int32 *degrees_prefix_sum =
            cst_dev_params.d_main_q_degrees_prefix_sum.channel(ichannel);
        main_q_idx = binsearch_maxle(degrees_prefix_sum, main_q_arc_index,
                                     main_q_offset, main_q_end - 1);

        // state_first_arc_idx_in_main_q
        // d_main_q_degrees_prefix_sum contains the prefix sum of the
        // degrees of all tokens in the main_q
        // d_main_q_degrees_prefix_sum[main_q_idx] contains the number of arc
        // in the main_q until that token
        const int32 state_first_arc_idx_in_main_q =
            degrees_prefix_sum[main_q_idx];

        // arc_offset_start is the offset in the CSR, to find the arcs
        // related to the state main_q_state_[main_q_idx]
        // it was set by the preprocess kernel
        const int32 arc_offset_start =
            cst_dev_params.d_main_q_arc_offsets.channel(ichannel)[main_q_idx];

        // local_arc_index is the arc index for that state
        // if local_arc_index == 2, we will process the second arc
        // of state main_q_state_[main_q_idx]
        const int32 local_arc_index =
            main_q_arc_index - state_first_arc_idx_in_main_q;

        // corresponding arc_idx in the FST
        arc_idx = arc_offset_start + local_arc_index;

        // Destination of that arc
        arc_next_state = cst_dev_params.d_arc_nextstates[arc_idx];

        // Building the total cost incrementally
        // we'll add the acoustic cost and the old token's cost
        const CostType arc_fixed_cost = cst_dev_params.d_arc_weights[arc_idx];
        const CostType prev_token_cost = orderedIntToFloat(
            cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx]
                .y);
        CostType total_cost = prev_token_cost + arc_fixed_cost;
        const int32 prev_state =
            cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx]
                .x;
        if (IS_EMITTING) {
          const int32 arc_ilabel = cst_dev_params.d_arc_pdf_ilabels[arc_idx];
          CostType acoustic_cost = -lane_counters->loglikelihoods[arc_ilabel];
          total_cost += acoustic_cost;
        }
        int_total_cost = floatToOrderedInt(total_cost);

        // If the total_cost is too large compared to our cutoff (beam search)
        // then let's drop it
        const IntegerCostType int_cutoff = lane_counters->int_cutoff;
        if (int_total_cost >= int_cutoff) int_total_cost = INT_MAX;
      }

      // If int_total_cost < INT_MAX, it means that :
      // - this thread had a valid input (main_q_arc_index < total_narcs)
      // - the total_cost of the generated token is < cutoff
      // We will then add that new token in the output queue, aux_q
      // We need to know where to put that token in the aux_q
      // we'll first compute its index inside the CUDA block
      // the first valid output token in the CUDA block will have index 0,
      // the second index 1... We compute that using a prefix sum
      //
      // We also need to find the overall min cost in the CUDA block
      // a prefix sum is a scan operation, and a min a reduce operation
      // we can perform a reduce operation using a scan (using the last value)
      // we compute the prefix sum and the min in one scan, using the data
      // struct CostTypeAndInt
      const int32 has_successor = (int_total_cost < INT_MAX) ? 1 : 0;

      int2 int_cost_and_index = {int_total_cost, has_successor};
      BlockScan(sh_temp_storage_scan)
          .InclusiveScan(int_cost_and_index, int_cost_and_index, MinPlus());
      if (KALDI_CUDA_DECODER_IS_LAST_1D_THREAD()) {
        // We are in a divergent branch
        // This is the last thread. The last value of the inclusive scan is the
        // total
        const int32 total_successors_in_block = int_cost_and_index.y;
        // Requesting a spot of size total_successors_in_block in the aux_q

        // note:  using 2 atomics here to avoid adding another kernel
        // first request more space
        const int aux_q_index_block_offset = atomicAdd(
            &lane_counters->aux_q_requested, total_successors_in_block);

        // check for overflow in aux_q
        // We try to prevent an overflow from happening using an adaptive beam
        // (cf GetAdaptiveBeam)
        if (aux_q_index_block_offset + total_successors_in_block <
            cst_dev_params.aux_q_capacity) {
          // no overflow

          // grab the aux_q offset
          sh_aux_q_index_block_offset =
              atomicAdd(&lane_counters->aux_q_end, total_successors_in_block);

          // We are not overflowing the queue, updating the global values
            IntegerCostType global_min_int_cost = lane_counters->min_int_cost;
            IntegerCostType local_min_int_cost = int_cost_and_index.x;
            // if we found a lower min_cost, update the global value
            if (local_min_int_cost < global_min_int_cost) {
              global_min_int_cost = local_min_int_cost;
              atomicMin(&lane_counters->min_int_cost, global_min_int_cost);
              CostType beam =
                  orderedIntToFloat(adaptive_int_beam_with_validity_index.x);
              IntegerCostType new_int_cutoff = floatToOrderedInt(
                  orderedIntToFloat(local_min_int_cost) + beam);
              atomicMin(&lane_counters->int_cutoff, new_int_cutoff);
            }
            int32 beam_valid_until_idx =
                adaptive_int_beam_with_validity_index.y;
            if (aux_q_index_block_offset >= beam_valid_until_idx) {
              // This beam is no longer valid. Updating it
              UpdateAdaptiveBeam(
                  cst_dev_params, aux_q_index_block_offset, global_min_int_cost,
                  &adaptive_int_beam_with_validity_index, lane_counters);
            }
        } else {
          // sh_aux_q_index_block_offset is in shared memory
          // its value is currently invalid (overflow)
          // we set it to a special value and use it as a flag to broadcast
          // the fact that we have an overflow and that all threads should exit
          sh_aux_q_index_block_offset = cst_dev_params.aux_q_capacity;

          // Setting the flag for the host. It will be used to print a warning
          // to stderr
          lane_counters->q_overflow |= OVERFLOW_AUX_Q;

          // We do not jump to end_lane now, because only
          // the first thread (threadIdx.x == 0) is executing this
          // We wait until the end of the divergent branch
        }
      }

      // Sync'ing for two reasons :
      // - Broadcasting sh_aux_q_index_block_offset
      // - reusing sh_temp_storage (cf CUB's doc)
      __syncthreads();
      // The only case where we can have that condition met,
      // is if we detected an overflow if the previous lines
      if (sh_aux_q_index_block_offset == cst_dev_params.aux_q_capacity)
        goto end_lane;  // done for this lane
      //
      // If we're executing the following lines it means everything
      // is valid and we are not overflowing the aux_q
      //
      int_cost_and_index.y -= has_successor;  // we want the exclusive sum now
      const int32 aux_q_block_index = int_cost_and_index.y;
      const int32 aux_q_index = sh_aux_q_index_block_offset + aux_q_block_index;
      if (has_successor) {
        // We save the new token to the aux_q
        cst_dev_params.d_aux_q_state_and_cost.lane(ilane)[aux_q_index] = {
            arc_next_state, int_total_cost};
        // Index of the parent token
        // the parent is the token used as input (source of arc)
        // that parent is at index main_q_idx in the GPU memory
        // However, the main_q is emptied before processing a new frame
        // we need to add the offset related to the previous frames index
        // we add cst_dev_params.main_q_global_offset
        const int32 prev_token =
            lane_counters->main_q_global_offset + main_q_idx;
        assert(main_q_idx >= 0 && main_q_idx < cst_dev_params.main_q_capacity);
        cst_dev_params.d_aux_q_info.lane(ilane)[aux_q_index] = {prev_token,
                                                                arc_idx};
      }
    }
  end_lane:;  // ";" is an empty statement
  }
}

// post_expand_kernel
// Called after expand_arcs_kernel
// Takes care of what needs to be done after an expand_arcs_kernel
// execution. Mostly resetting the beam (if adaptive beam was triggered,
// the max_active_ kernels will take care of selecting a good beam),
// resetting the number of arcs in the main_q (we've processed them all),
// etc.
// Threads (1,1,1)
// Blocks (1, nlanes_used, 1)
template <bool IS_EMITTING>
__global__ void post_expand_kernel(DeviceParams cst_dev_params,
                                   KernelParams params) {
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    LaneCounters *h_lane_counters = cst_dev_params.h_lanes_counters.lane(ilane);
    const int prev_main_q_end = lane_counters->main_q_narcs_and_end.y;
    const int prev_n_extra_prev_tokens =
        lane_counters->main_q_n_extra_prev_tokens;
    const int aux_q_end = lane_counters->aux_q_end;
    CostType min_cost = orderedIntToFloat(lane_counters->min_int_cost);
    // The next step is the contracting step from aux_q to main_q
    // It will need the aux_q_end value. But it will also empty the aux_q
    // We're resetting aux_q_end to 0 now, but we're saving its old value
    // in another place
    lane_counters->post_expand_aux_q_end = aux_q_end;
    h_lane_counters->post_expand_aux_q_end = aux_q_end;       // pinned memory
    h_lane_counters->q_overflow = lane_counters->q_overflow;  // pinned memory
    lane_counters->aux_q_end = 0;
    lane_counters->aux_q_requested = 0;
    // We are done processing those arcs
    lane_counters->main_q_narcs_and_end.x = 0;
    // Resetting the adaptive beam
    lane_counters->adaptive_int_beam_with_validity_index.x =
        lane_counters->int_beam;
    lane_counters->adaptive_int_beam_with_validity_index.y =
        cst_dev_params.adaptive_beam_static_segment;
    CostType beam = orderedIntToFloat(lane_counters->int_beam);
    lane_counters->int_cutoff = floatToOrderedInt(min_cost + beam);
    // If the adaptive beam kicked in, we want to reset the beam
    // the max-active process will take care of selecting the right beam
    if (IS_EMITTING) {
      // the main_q contains the tokens from the previous frame
      // after emitting, we won't use them anymore to create new tokens
      // we reset the main_q
      lane_counters->main_q_narcs_and_end = {0, 0};
      lane_counters->main_q_requested = 0;
      // The main_q was flushed - we need to update the global_offset
      lane_counters->main_q_global_offset += prev_main_q_end;
      if (threadIdx.x == 0 && blockIdx.x == 0)
        lane_counters->main_q_extra_prev_tokens_global_offset +=
            prev_n_extra_prev_tokens;
      // Moving local offset. Tokens created by last expand
      // will be pruned, and survivals will be moved at the end
      // of the main q. Those tokens will be placed after local_offset
      lane_counters->main_q_requested = 0;
      CostType min_cost = orderedIntToFloat(lane_counters->min_int_cost);
      lane_counters->min_histo_cost = min_cost;
      lane_counters->max_histo_cost = min_cost + beam;
      lane_counters->histo_bin_width = beam / (KALDI_CUDA_DECODER_HISTO_NBINS-1);
    } else {
      lane_counters->main_q_local_offset = prev_main_q_end;
      // reset requested to end of queue
      lane_counters->main_q_requested = prev_main_q_end;
    }
  }
}

__global__ void post_contract_and_preprocess_kernel(DeviceParams cst_dev_params,
                                                    KernelParams params) {
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    LaneCounters *h_lane_counters = cst_dev_params.h_lanes_counters.lane(ilane);
    int2 main_q_narcs_and_end = lane_counters->main_q_narcs_and_end;
    h_lane_counters->main_q_narcs_and_end =
        main_q_narcs_and_end;                                 // pinned memory
    h_lane_counters->q_overflow = lane_counters->q_overflow;  // pinned memory
    atomicMin(&lane_counters->main_q_n_emitting_tokens, main_q_narcs_and_end.y);
  }
}

// Meta-kernel (merging preprocess and expand) but only works with 1 CUDA block
// Used to avoid calling multiple main kernels (such as expand_arcs_kernel)
// for the tail of non emitting (lots of iterations with small number of arcs)
//
// Code is greatly simplified because we use only one CTA / lane
//
// Repeat until new queue empty:
// 1) Preprocess
// 2) Expand arcs
//
// The preprocess stage is not done on the first iteration, because it was
// already done by the ProcessAndContract kernel. We always call
// PruneAndPreprocess before calling FinalizeProcessNonemitting
//
// At the end, this kernel finalize the computation for current frame,
// so that it's ready for next ProcessEmitting
//
// This kernel works, but can be greatly simplified now.
__launch_bounds__(KALDI_CUDA_DECODER_LARGEST_1D_BLOCK, 1) __global__
    void finalize_process_non_emitting_kernel(DeviceParams cst_dev_params,
                                              KernelParams params) {
  typedef cub::BlockScan<int2, KALDI_CUDA_DECODER_LARGEST_1D_BLOCK>
      Int2BlockScan;
  typedef cub::BlockScan<int, KALDI_CUDA_DECODER_LARGEST_1D_BLOCK> IntBlockScan;
  __shared__ typename IntBlockScan::TempStorage sh_temp_storage_int_scan;
  __shared__ typename Int2BlockScan::TempStorage sh_temp_storage_int2_scan;

  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    const int32 ichannel = lane_counters->channel_to_compute;
    ChannelCounters *channel_counters =
        cst_dev_params.d_channels_counters.channel(ichannel);

    int2 both = lane_counters->main_q_narcs_and_end;
    int32 main_q_narcs = both.x;
    int32 main_q_end = both.y;
    int32 main_q_local_offset = lane_counters->main_q_local_offset;
    const int32 main_q_global_offset = lane_counters->main_q_global_offset;
    // aux_q is empty when this kernel is called
    int32 aux_q_end = 0;
    IntegerCostType int_cutoff = lane_counters->int_cutoff;
    while (main_q_narcs > 0) {
      // Step 1 : ExpandArcs
      KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(offset, thread_idx,
                                                     main_q_narcs) {
        const int32 main_q_arc_idx = offset + thread_idx;
        // For details on how this code works, please refer to comments in
        // expand_arcs
        IntegerCostType total_int_cost = INT_MAX;
        int32 arc_idx;
        StateId arc_next_state;
        int32 main_q_idx;
        if (main_q_arc_idx < main_q_narcs) {
          main_q_idx = binsearch_maxle(
              cst_dev_params.d_main_q_degrees_prefix_sum.channel(ichannel),
              main_q_arc_idx, main_q_local_offset, main_q_end - 1);

          const int32 state_first_arc_idx_in_main_q =
              cst_dev_params.d_main_q_degrees_prefix_sum.channel(
                  ichannel)[main_q_idx];
          const int32 arc_offset_start =
              cst_dev_params.d_main_q_arc_offsets.channel(ichannel)[main_q_idx];
          arc_idx = arc_offset_start +
                    (main_q_arc_idx - state_first_arc_idx_in_main_q);

          arc_next_state = cst_dev_params.d_arc_nextstates[arc_idx];
          CostType arc_weight = cst_dev_params.d_arc_weights[arc_idx];
          CostType prev_token_cost =
              orderedIntToFloat(cst_dev_params.d_main_q_state_and_cost
                                    .channel(ichannel)[main_q_idx]
                                    .y);
          total_int_cost = floatToOrderedInt(arc_weight + prev_token_cost);
	  if(total_int_cost < lane_counters->min_int_cost)
            atomicMin(&lane_counters->min_int_cost, total_int_cost);
          if (total_int_cost >= int_cutoff) {
            total_int_cost = INT_MAX;  // above cutoff
          }
        }
        const int32 has_successor = (total_int_cost < INT_MAX) ? 1 : 0;

        int32 local_aux_q_idx;
        int32 nsuccessors;
        IntBlockScan(sh_temp_storage_int_scan)
            .ExclusiveSum(has_successor, local_aux_q_idx,
                          nsuccessors);  // aggregate

        // Checking if we are overflowing the aux_q
        if ((aux_q_end + nsuccessors) >= cst_dev_params.aux_q_capacity) {
          lane_counters->q_overflow |= OVERFLOW_AUX_Q;
          // nothing to revert in global memory
          goto finalize_lane;
        }

        if (has_successor) {
          const int32 aux_q_idx = aux_q_end + local_aux_q_idx;
          const int32 prev_token_idx = main_q_global_offset + main_q_idx;
          cst_dev_params.d_aux_q_state_and_cost.lane(ilane)[aux_q_idx] = {
              arc_next_state, total_int_cost};
          cst_dev_params.d_aux_q_info.lane(ilane)[aux_q_idx] = {prev_token_idx,
                                                                arc_idx};
        }
        aux_q_end += nsuccessors;
        // sync: reusing sh_temp_storage_scan_int
        __syncthreads();
      }

      // Step 2 : PreprocessAndContract
      // Reset for new iteration
      main_q_narcs = 0;
      main_q_local_offset = main_q_end;
      KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(offset, thread_idx,
                                                     aux_q_end) {
        const int32 aux_q_idx = offset + thread_idx;
        int32 degree = 0;
        int32 start = -1;
        StateId token_state;
        IntegerCostType token_int_cost;
        if (aux_q_idx < aux_q_end) {
          int2 both =
              cst_dev_params.d_aux_q_state_and_cost.lane(ilane)[aux_q_idx];
          token_state = both.x;
          token_int_cost = both.y;
          // beam may have changed since generation
          // We are non-emitting in this kernel, using ne offsets
          start = cst_dev_params.d_arc_ne_offsets[token_state];
          int32 end = cst_dev_params.d_arc_ne_offsets[token_state + 1];
          degree = end - start;
        }
        int has_valid_nonpruned_token = (start != -1) ? 1 : 0;
        int2 narcs_and_ntokens_prefix_sum = {degree, has_valid_nonpruned_token};
        int2 aggregate, zero2 = {0, 0};
        Int2BlockScan(sh_temp_storage_int2_scan)
            .ExclusiveScan(narcs_and_ntokens_prefix_sum,
                           narcs_and_ntokens_prefix_sum, zero2, PlusPlus(),
                           aggregate);
        // Checking if we are not overflowing the main_q
        const int32 total_ntokens = aggregate.y;
        if ((main_q_end + total_ntokens) >= cst_dev_params.main_q_capacity) {
          lane_counters->q_overflow |= OVERFLOW_MAIN_Q;
          goto finalize_lane;
        }
        const int32 degree_prefix_sum =
            main_q_narcs + narcs_and_ntokens_prefix_sum.x;
        const int32 degree_sum = aggregate.x;
        main_q_narcs += degree_sum;
        if (has_valid_nonpruned_token) {
          const int32 local_main_q_idx = narcs_and_ntokens_prefix_sum.y;
          const int32 main_q_idx = main_q_end + local_main_q_idx;

          cst_dev_params.d_main_q_arc_offsets.channel(ichannel)[main_q_idx] =
              start;
          cst_dev_params.d_main_q_degrees_prefix_sum.channel(
              ichannel)[main_q_idx] = degree_prefix_sum;
          cst_dev_params.d_main_q_state_and_cost.channel(
              ichannel)[main_q_idx] = {token_state, token_int_cost};
          cst_dev_params.d_main_q_info.lane(ilane)[main_q_idx] =
              cst_dev_params.d_aux_q_info.lane(ilane)[aux_q_idx];
          cst_dev_params.d_main_q_acoustic_cost.lane(ilane)[main_q_idx] =
              0.0f;  // we are always nonemitting in this kernel
        }
        main_q_end += total_ntokens;
        __syncthreads();
      }
      aux_q_end = 0;  // aux_q is now empty
    }

  finalize_lane:
    if (threadIdx.x == 0) {
      // This main_q is now final for that frame
      lane_counters->main_q_narcs_and_end = {0, main_q_end};
      cst_dev_params.h_lanes_counters.lane(ilane)->main_q_narcs_and_end = {
          0, main_q_end};  // pinned memory
    }
  }
}

// GetBestCost :
// Finds all tokens with a cost in [min_cost;min_cost+lattice_beam[
// Add the final_costs if use_final_probs
// Does the computation in two steps
//
// Step 1: Find the value of min_cost, i.e. the minimum cost in the last token
// queue
// (the queue generated by the last frame computed)
// We set both channel_counters->min_int_cost_and_arg_without_final
// and channel_counters->min_int_cost_and_arg_with_final
// One add the final_cost[token.state] before looking for the min
__global__ void get_best_cost_step1_kernel(DeviceParams cst_dev_params,
                                           KernelParams params,
                                           bool use_final_probs,
                                           CostType fst_zero) {
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    const int32 ichannel = lane_counters->channel_to_compute;
    ChannelCounters *channel_counters =
        cst_dev_params.d_channels_counters.channel(ichannel);
    const int32 main_q_end = channel_counters->prev_main_q_narcs_and_end.y;
    const int32 global_offset = channel_counters->prev_main_q_global_offset;
    KALDI_CUDA_DECODER_1D_KERNEL_LOOP(idx, main_q_end) {
      if (idx == 0)
        lane_counters->n_within_lattice_beam =
            0;  // will be used in the next kernel
      const int2 both =
          cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[idx];
      const int token_state = both.x;
      const int token_int_cost = both.y;
      CostType cost = orderedIntToFloat(token_int_cost);
      IntegerCostType int_cost = floatToOrderedInt(cost);
      int32 global_idx = global_offset + idx;
      // We know what is the min cost (without final costs)
      // we just need to have the index of one token with that min cost

      if (use_final_probs) {
        const CostType final_cost =
            cst_dev_params.d_fst_final_costs[token_state];
        IntegerCostType int_cost_with_final =
            floatToOrderedInt(cost + final_cost);
        if (final_cost != fst_zero) {
          int2 min_and_arg = {int_cost_with_final,
                              global_idx};  // sort by cost, put it first
          atomicMinI2(&channel_counters->min_int_cost_and_arg_with_final,
                      min_and_arg);
        }
      }
    }
  }
}

// Step2: Now that step1 found the min_cost (with and without final cost)
// If at least one final token (token associated with a final fst state)
// exists in the token queue, AND if use_final_probs is true,
// We can detect all tokens with a cost within [min_cost;min_cost+lattice_beam]
// and list them into d_list_final_tokens_in_main_q
__global__ void get_best_cost_step2_kernel(DeviceParams cst_dev_params,
                                           KernelParams params,
                                           bool use_final_probs,
                                           CostType fst_zero) {
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    const int32 ichannel = lane_counters->channel_to_compute;
    const ChannelCounters *channel_counters =
        cst_dev_params.d_channels_counters.channel(ichannel);
    const int32 main_q_end = channel_counters->prev_main_q_narcs_and_end.y;
    const int32 global_offset = channel_counters->prev_main_q_global_offset;
    const int2 min_int_cost_and_arg_with_final =
        channel_counters->min_int_cost_and_arg_with_final;
    const int2 min_int_cost_and_arg_without_final =
        channel_counters->min_int_cost_and_arg_without_final;
    bool has_reached_final = (min_int_cost_and_arg_with_final.x != INT_MAX);
    // Use final if we want to use final (use_final_probs is true) and if we
    // found a final state in the token list
    bool compute_final = use_final_probs && has_reached_final;
    IntegerCostType min_cost_to_use =
        compute_final ? min_int_cost_and_arg_with_final.x
                      : min_int_cost_and_arg_without_final.x;

    // if token.cost < lattice_cutoff, that token will belong in the output
    // lattice
    CostType lattice_cutoff =
        orderedIntToFloat(min_cost_to_use) + cst_dev_params.lattice_beam;
    IntegerCostType lattice_int_cutoff = floatToOrderedInt(lattice_cutoff);
    KALDI_CUDA_DECODER_1D_KERNEL_LOOP(idx, main_q_end) {
      // First thread of each lane will move the results into lane counters.
      // That's because we never move channel counters back to host,
      // so we move those values to the lane counters, and those lane counters
      // will be moved to host after this kernel
      if (idx == 0) {
        // The lane counters will be copied to host
        lane_counters->min_int_cost_and_arg =
            compute_final ? min_int_cost_and_arg_with_final
                          : min_int_cost_and_arg_without_final;
        lane_counters->has_reached_final = has_reached_final;
      }
      // Looking for a token with its int_cost < lattice_int_cutoff
      const int2 both =
          cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[idx];
      const int32 token_state = both.x;
      int32 token_int_cost = both.y;
      if (compute_final) {
        const CostType final_cost =
            cst_dev_params.d_fst_final_costs[token_state];
        const CostType token_cost = orderedIntToFloat(token_int_cost);
        // final_cost == fst_zero -> this state is not final
        token_int_cost = (final_cost != fst_zero)
                             ? floatToOrderedInt(token_cost + final_cost)
                             : INT_MAX;
      }
      if (token_int_cost < lattice_int_cutoff) {
        // That token will be included in the lattice (last frame)
        // save it
        int list_idx = atomicAdd(&lane_counters->n_within_lattice_beam, 1);
        cst_dev_params.h_list_final_tokens_in_main_q.lane(ilane)[list_idx] = {
            global_offset + idx, token_int_cost};
      }
    }
  }
}
__global__ void get_best_cost_step3_kernel(DeviceParams cst_dev_params,
                                           KernelParams params) {
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *d_lanes_counters =
        cst_dev_params.d_lanes_counters.lane(ilane);
    LaneCounters *h_lanes_counters =
        cst_dev_params.h_lanes_counters.lane(ilane);
    h_lanes_counters->min_int_cost_and_arg =
        d_lanes_counters->min_int_cost_and_arg;
    h_lanes_counters->has_reached_final = d_lanes_counters->has_reached_final;
    h_lanes_counters->n_within_lattice_beam =
        d_lanes_counters->n_within_lattice_beam;
  }
}
// compute_costs_histogram_kernel
// Used in ApplyMaxActiveAndReduceBeam
// Compute the histogram of the token.cost in the main_q
__global__ void compute_costs_histogram_kernel(DeviceParams cst_dev_params,
                                               KernelParams params,
                                               bool use_aux_q) {
  const int nlanes = params.nlanes_used;
  typedef cub::BlockHistogram<BinId, KALDI_CUDA_DECODER_1D_BLOCK, 1,
                              KALDI_CUDA_DECODER_HISTO_NBINS + 1>
      BlockHistogram;
  __shared__ typename BlockHistogram::TempStorage temp_storage;
  __shared__ unsigned int smem_histogram[KALDI_CUDA_DECODER_HISTO_NBINS + 1];

  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    const int32 ichannel = lane_counters->channel_to_compute;
    const int32 q_end = use_aux_q ? lane_counters->post_expand_aux_q_end
                                  : lane_counters->main_q_narcs_and_end.y;
    bool compute_max_active = lane_counters->compute_max_active;
    if (!compute_max_active) {
      if (q_end <= cst_dev_params.max_active) continue;  // nothing to do
      // Otherwise let's turn max active on for this frame and lane
      lane_counters->compute_max_active = true;
    }

    // Reset local histogram for this lane
    BlockHistogram(temp_storage).InitHistogram(smem_histogram);
    CostType min_histo_cost = lane_counters->min_histo_cost;
    CostType max_histo_cost = lane_counters->max_histo_cost;
    CostType bin_width = lane_counters->histo_bin_width;

    // We have a sync inside the loop, keeping all threads alive
    KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(block_offset, thread_idx,
                                                   q_end) {
      const int32 q_idx = block_offset + thread_idx;
      // The last bin is for everything we don't want to count:
      // cost already above the beam, or non-valid tokens
      // It is the default bin
      BinId bin_id[1];
      bin_id[0] = KALDI_CUDA_DECODER_HISTO_NBINS;
      if (q_idx < q_end) {
        IntegerCostType int_cost =
            use_aux_q
                ? cst_dev_params.d_aux_q_state_and_cost.lane(ilane)[q_idx].y
                : cst_dev_params.d_main_q_state_and_cost
                      .channel(ichannel)[q_idx]
                      .y;
        CostType cost = orderedIntToFloat(int_cost);
        CostType extra = cost - min_histo_cost;
	if(extra <= 0.0f) 
		bin_id[0] = 0;
  	else if (extra < max_histo_cost) {
          bin_id[0] = (BinId)__fdiv_rd(extra, bin_width)+1; // +1 because first bin is cost < min_histo_cost
        }
      }
      BlockHistogram(temp_storage).Composite(bin_id, smem_histogram);  // sync
      __syncthreads();  // reusing temp_storage
    }

    // Not using the macros 1D_LOOP because that loop is only within a CTA
    for (int32 bin_id_w = threadIdx.x;
         bin_id_w < KALDI_CUDA_DECODER_HISTO_NBINS;
         bin_id_w += KALDI_CUDA_DECODER_1D_BLOCK) {
      // Writing the local histo to global
      // We don't care about the last bin (cf above)
      int32 s_count = (int32)smem_histogram[bin_id_w];
      atomicAdd(&cst_dev_params.d_histograms.lane(ilane)[bin_id_w], s_count);
    }
    // Making sure we're done reading from smem
    __syncthreads();
  }
}

// update_beam_using_histogram_kernel
// used in ApplyMaxActiveAndReduceBeam
// uses the histogram computed in compute_costs_histogram_kernel
// to find where to cut (where to set the beam)
// to keep only ~max_active_ tokens.
// Important: use only one CTA per lane
__global__ void update_beam_using_histogram_kernel(DeviceParams cst_dev_params,
                                                   KernelParams params,
                                                   bool use_aux_q) {
  typedef cub::BlockScan<int, KALDI_CUDA_DECODER_1D_BLOCK> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const int nlanes = params.nlanes_used;
  const int max_active = cst_dev_params.max_active;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    bool compute_max_active = lane_counters->compute_max_active;
    if (!compute_max_active) continue;  // nothing to do
    CostType beam = orderedIntToFloat(lane_counters->int_beam);
    CostType min_histo_cost = lane_counters->min_histo_cost;
    CostType bin_width = lane_counters->histo_bin_width;
    // We now have our histogram of the token costs (computed in the previous
    // kernel)
    // Each thread i is responsible for a bin i, with that bin containing ni
    // tokens.
    // We compute the prefix sum of those ni, ending up for each thread with
    // si=sum[i=1..i](ni)
    // If the thread i detects that si < max_active_ and s[i+1] >= max_active_,
    // then we will cut the beam at
    // the cost of the bin [i+1]
    //
    // Assert : one thread in a CTA is responsible for at most one bin
    // we will not iterate over bins
    assert(KALDI_CUDA_DECODER_HISTO_NBINS < KALDI_CUDA_DECODER_1D_BLOCK);
    int bin_id = threadIdx.x;
    int val = 0;
    if (bin_id < KALDI_CUDA_DECODER_HISTO_NBINS) 
      val = cst_dev_params.d_histograms.lane(ilane)[bin_id];
    
    int prefix_sum;
    BlockScan(temp_storage).ExclusiveSum(val, prefix_sum);

    if (prefix_sum < max_active && (prefix_sum + val) >= max_active) {
      // We found our new beam regarding min_histo_cost
      // Howevever, the current min_cost could be lower than min_histo_cost
      // we need to add that diff to the new beam
      CostType new_beam_for_histo_min_cost = bin_width * bin_id;
      CostType current_min_cost = orderedIntToFloat(lane_counters->min_int_cost);
      CostType new_beam = (min_histo_cost - current_min_cost) + new_beam_for_histo_min_cost;
      IntegerCostType new_int_beam = floatToOrderedInt(new_beam);
      // Saving our new beam for this lane
      lane_counters->int_beam = new_int_beam;
      lane_counters->adaptive_int_beam_with_validity_index.x = new_int_beam;
      lane_counters->int_cutoff = floatToOrderedInt(current_min_cost + new_beam);
    }
  }
}

//
// PostProcessingMainQueue kernels.
// all the following kernels are called when postprocessing a frame
//

// Filling hashmap values with the tokens that we have in the main queue
// We do that because multiple tokens associated with the same FST state
// (but with different arc_idx) can exist in the main_q. We need to detect
// that situation, count them, detect what the min_cost for that FST state is.
// It is done using a hashmap
__global__ void fill_hashmap_with_main_q_kernel(DeviceParams cst_dev_params,
                                                KernelParams params) {
  // Operator for the prefix sum inside the CUDA block
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    const int32 ichannel = lane_counters->channel_to_compute;
    ChannelCounters *channel_counters =
        cst_dev_params.d_channels_counters.channel(ichannel);

    const int32 main_q_end = lane_counters->main_q_narcs_and_end.y;
    int32 min_int_cost = lane_counters->min_int_cost;
    CostType min_cost = orderedIntToFloat(min_int_cost);
    const int32 global_offset = channel_counters->prev_main_q_global_offset;
    KALDI_CUDA_DECODER_1D_KERNEL_LOOP(main_q_idx, main_q_end) {
      // Position of considered token in the main_q
      if (main_q_idx < main_q_end) {
        int2 both = cst_dev_params.d_main_q_state_and_cost.channel(
            ichannel)[main_q_idx];
        StateId token_state = both.x;
        IntegerCostType token_int_cost = both.y;
        if (min_int_cost == token_int_cost) {
          // remove offset = min_cost, set it to 0 explicitely
          token_int_cost = floatToOrderedInt(0.0f);
          channel_counters->min_int_cost_and_arg_without_final = {
              token_int_cost, global_offset + main_q_idx};
          lane_counters->prev_arg_min_int_cost = main_q_idx;
        } else {
          // remove offset = min_cost
          CostType token_cost = orderedIntToFloat(token_int_cost) - min_cost;
          token_int_cost = floatToOrderedInt(token_cost);
        }
        int local_idx, hash_idx;
        hashmap_insert_or_aggregate(cst_dev_params.d_hashmap_values.lane(ilane),
                                    token_state, token_int_cost, main_q_idx,
                                    cst_dev_params.hashmap_capacity, &local_idx,
                                    &hash_idx);
        cst_dev_params.d_main_q_n_extra_prev_tokens_local_idx.lane(
            ilane)[main_q_idx] = local_idx;
        cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx].y =
            token_int_cost;
        // If we have the min, saving its index for get best cost and the min
        // cost estimate of the next frame

        // Saving where that token.state ended up in the hashmap
        // false = this token is not the representative of this state
        // We will update representing_state once we know more (in the next
        // kernel)
        // We first need to add all tokens to the hashmap. Which will be the
        // case when
        // this kernel returns.
        SetFSTStateHashIndex(
            hash_idx, false,
            &cst_dev_params.d_main_q_state_hash_idx.lane(ilane)[main_q_idx]);
      }

      if (main_q_idx == 0) {
        lane_counters->int_cutoff = floatToOrderedInt(
            orderedIntToFloat(lane_counters->int_cutoff) - min_cost);
      }
    }
  }
}

// preprocess_and_list_extra_prev_tokens_kernel_step[i] kernels
// Called in PostProcessingMainQueue
// They do two things:
// - do the "emitting preprocessing". I.e. doing the preprocessing necessary for
// the future ExpandArcsEmitting that may be done next (if the current frame is
// not the last one)
// It consists of filling the d_main_q_degrees_prefix_sum of the emitting arc
// degrees of the tokens + setting d_main_q_arc_offsets
// - when we have multiple tokens associated with the same FST state S, we will
// list them in d_main_q_extra_prev_tokens. We need to know where to put them in
// that array,
// so we'll compute a prefix_sum also to compute those indexes. We'll then save
// the location of each extra tokens list (its offset and size in
// d_main_q_extra_prev_tokens),
// and save it into d_main_q_info for later lattice processing
//
// First step : Reading the hashmap, detecting which token is representative for
// each FST state, which is decided by fill_hashmap_with_main_q_kernel()
// (we pick one of the best ones, with the best ones being the ones with the
// lowest cost)
// this representative will be responsible for K tokens, with K being the number
// of tokens associated with that FST state. We only considers the cases where K
// > 1,
// because if K == 1, then we will not store that token in the special list
// d_main_q_extra_prev_tokens
// Each representative is also the only token that will propagate emitting arcs
// for that FST state. Because a representative has the min_cost for that FST
// state, it is enough to only propagate
// that one
// Each representative counts the number of emitting arcs it is responsible for,
// and we will compute the prefix sum of the arc degrees
__global__ void emitting_preprocess_and_list_extra_prev_tokens_step1_kernel(
    DeviceParams cst_dev_params, KernelParams params) {
  // Operator for the prefix sum inside the CUDA block
  typedef cub::BlockScan<int2, KALDI_CUDA_DECODER_1D_BLOCK> BlockScan;
  __shared__ typename BlockScan::TempStorage sh_temp_storage;
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    const LaneCounters *lane_counters =
        cst_dev_params.d_lanes_counters.lane(ilane);
    const int32 main_q_end = lane_counters->main_q_narcs_and_end.y;
    // Final cutoff from last ExpandArc execution
    // The cutoff can have decreased since moving tokens to the main_q
    // min_cost cannot be lower than before (we only did non-emitting phases
    // since then)
    // but the adaptive beam may have lowered the beam
    const IntegerCostType int_cutoff = lane_counters->int_cutoff;
    // Keeping all threads in CTA alive
    // We'll __syncthreads()
    KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(block_offset, thread_idx,
                                                   main_q_end) {
      // We'll take care of the token at index main_q_idx
      const int32 main_q_idx = block_offset + thread_idx;
      const int32 ichannel = lane_counters->channel_to_compute;
      // If that token is the representative of its FST state (token.next_state)
      // The representative of a FST state is the token with the lowest
      // token.cost for that FST state
      // If multiple tokens have token1.cost == token2.cost ==
      // min_cost_for_that_state, then one is picked (first come first serve,
      // was done in fill_hashmap_with_main_q_kernel)
      bool representing_state = false;
      // Number of emitting arcs for that token
      // Only the token representative of that FST state can have degree > 0
      int32 degree = 0;
      // If that token is representative of a FST state S,
      // and if multiple tokens are associated with that state S,
      // then n_extra_prev_token will contain their count
      int32 n_extra_prev_token = 0;
      if (main_q_idx < main_q_end) {
        int2 both = cst_dev_params.d_main_q_state_and_cost.channel(
            ichannel)[main_q_idx];
        StateId token_state = both.x;
        IntegerCostType token_int_cost = both.y;
        // Loading info about token.next_state. Is there multiple tokens for
        // that state ?
        // How many ? What's the min token.cost for that state ?
        int32 hash_idx;    // we saved the hash_idx after inserting
        bool bool_buffer;  // will always be false. We just need it to call the
                           // function
        GetFSTStateHashIndex(
            cst_dev_params.d_main_q_state_hash_idx.lane(ilane)[main_q_idx],
            &hash_idx, &bool_buffer);
        HashmapValueT h_val =
            cst_dev_params.d_hashmap_values.lane(ilane)[hash_idx];
        // Token index of one of the token which the lowest token.cost for that
        // state
        uint32_t state_best_int_cost_argmin;
	GetArgFromPackedArgminUInt64(h_val.min_and_argmin_int_cost_u64, &state_best_int_cost_argmin);

        // Checking if we're the representative of that state
        representing_state = (main_q_idx == state_best_int_cost_argmin);
        // Saving the hash_idx of that fst state + if we're responsible for that
        // state
        SetFSTStateHashIndex(
            hash_idx, representing_state,
            &cst_dev_params.d_main_q_state_hash_idx.lane(ilane)[main_q_idx]);

        // One of the best token for that state will represent that state in the
        // next frame
        if (representing_state) {
          if (token_int_cost < int_cutoff) {
            // Next step is emitting (next frame), using emitting offsets
            const int32 start = cst_dev_params.d_arc_e_offsets[token_state];
            const int32 end = cst_dev_params.d_arc_e_offsets[token_state + 1];
            degree = end - start;
            // Saving the start offset for the expand kernel
            // avoid a new random memory access
            cst_dev_params.d_main_q_arc_offsets.channel(ichannel)[main_q_idx] =
                start;
          }
          // If that FST state has only one token associated to it, we store
          // that token directly in
          // d_main_q_info (its original place)
          // We only move it into the d_main_q_extra_prev_tokens list if
          // multiple tokens are associated to that state
          n_extra_prev_token = (h_val.count > 1) ? (h_val.count) : 0;
        }
      }

      // Computing a local prefix sum inside that CUDA block
      // Others kernels will take care of adding the necessary offset to those
      // local prefix sums
      int2 zeroi2 = {0, 0};
      int2 vali2 = {degree, n_extra_prev_token};
      int2 aggi2;
      BlockScan(sh_temp_storage)
          .ExclusiveScan(vali2, aggi2, zeroi2, PlusPlus());
      int32 degree_local_prefix_sum = aggi2.x;
      int32 n_extra_prev_token_prefix_sum = aggi2.y;

      if (main_q_idx < main_q_end) {
        // This is not the final global prefix sum
        // Other kernels will add the necessary offset
        cst_dev_params.d_main_q_degrees_prefix_sum.channel(
            ichannel)[main_q_idx] = degree_local_prefix_sum;
        cst_dev_params.d_main_q_extra_prev_tokens_prefix_sum.lane(
            ilane)[main_q_idx] = n_extra_prev_token_prefix_sum;
      }

      if (KALDI_CUDA_DECODER_IS_LAST_1D_THREAD()) {
        // Saving the local sum of degrees of that CUDA block
        // That's necessary to compute the global offset of that CUDA block,
        // and that offset is what we need to transform the local prefix sum
        // into a global prefix sum
        const int local_sum_index = block_offset / KALDI_CUDA_DECODER_1D_BLOCK;
        // the prefix sum was exclusive, adding missing value
        const int degree_inclusive_sum = degree_local_prefix_sum + degree;
        const int n_extra_prev_tokens_inclusive_sum =
            n_extra_prev_token_prefix_sum + n_extra_prev_token;
        cst_dev_params.d_main_q_block_sums_prefix_sum.lane(
            ilane)[local_sum_index] = {degree_inclusive_sum,
                                       n_extra_prev_tokens_inclusive_sum};
      }

      // Synchronization because:
      // - we may need to reuse sh_temp_storage if the for loop iterates (cf
      // CUB's doc)
      __syncthreads();
    }
  }
}

// In step1, we've computed the local (CTA-wide) prefix sums. We also have the
// local sums of each individual CTAs
// In this kernel, we will compute the offset of each CTA in the global prefix
// sum. We will then add those offsets in step3
// Only one CTA / lane
__global__ void emitting_preprocess_and_list_extra_prev_tokens_step2_kernel(
    DeviceParams cst_dev_params, KernelParams params) {
  typedef cub::BlockScan<int2, KALDI_CUDA_DECODER_1D_BLOCK> BlockScan;
  __shared__ typename BlockScan::TempStorage sh_temp_storage;
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    const int main_q_end = lane_counters->main_q_narcs_and_end.y;
    const int ntiles = KALDI_CUDA_DECODER_DIV_ROUND_UP(
        main_q_end, KALDI_CUDA_DECODER_1D_BLOCK);
    // Using block_offset loop to keep entire CTA alive (we're going to use
    // __syncthreads in CUB)
    int2 sum_so_far = {0, 0};
    KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(offset, thread_idx, ntiles) {
      const int32 itile = offset + thread_idx;
      const int2 zeroi2 = {0, 0};
      const int2 val =
          (itile < ntiles)
              ? cst_dev_params.d_main_q_block_sums_prefix_sum.lane(ilane)[itile]
              : zeroi2;

      int2 prefix_sum, sum;
      BlockScan(sh_temp_storage)
          .ExclusiveScan(val, prefix_sum, zeroi2, PlusPlus(), sum);
      PlusPlus pp;
      prefix_sum = pp(prefix_sum, sum_so_far);
      sum_so_far = pp(sum_so_far, sum);
      if (itile < ntiles) {
        cst_dev_params.d_main_q_block_sums_prefix_sum.lane(ilane)[itile] =
            prefix_sum;
      }
      if (itile == (ntiles - 1)) {
        const int32 total_narcs = prefix_sum.x + val.x;
        const int32 total_n_extra_prev_tokens = prefix_sum.y + val.y;
        lane_counters->main_q_narcs_and_end.x = total_narcs;
        lane_counters->main_q_n_extra_prev_tokens = total_n_extra_prev_tokens;
        assert(total_n_extra_prev_tokens >= 0 &&
               total_n_extra_prev_tokens <= main_q_end);
      }
    }
  }
}

// Step3: Uses the CTA offsets computed in step2 to transform the CTA-wide
// prefix sums to global prefix sums
// The representative of each FST states saves into the hashmap the location of
// the extra_prev_tokens of that state
// in d_main_q_extra_prev_tokens. That way each extra tokens will know where to
// write itself in the next kernel.
__global__ void emitting_preprocess_and_list_extra_prev_tokens_step3_kernel(
		DeviceParams cst_dev_params, KernelParams params) {
	const int nlanes = params.nlanes_used;
	KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
		const LaneCounters *lane_counters =
			cst_dev_params.d_lanes_counters.lane(ilane);
		const int32 ichannel = lane_counters->channel_to_compute;
		const int main_q_end = lane_counters->main_q_narcs_and_end.y;
		KALDI_CUDA_DECODER_1D_KERNEL_LOOP(main_q_idx, main_q_end) {
			const int32 local_sum_idx = main_q_idx / KALDI_CUDA_DECODER_1D_BLOCK;
			const int2 local_sum_offset =
				cst_dev_params.d_main_q_block_sums_prefix_sum.lane(
						ilane)[local_sum_idx];
			cst_dev_params.d_main_q_degrees_prefix_sum.channel(
					ichannel)[main_q_idx] += local_sum_offset.x;
			int extra_prev_tokens_offset =
				cst_dev_params.d_main_q_extra_prev_tokens_prefix_sum.lane(
						ilane)[main_q_idx] +
				local_sum_offset.y;
			// Loading the hash index associate with token.state
			// If representative, store the location of the extra prev tokens list for
			// that state in the hashmap
			bool is_representative;
			int32 hash_idx;
			GetFSTStateHashIndex(
					cst_dev_params.d_main_q_state_hash_idx.lane(ilane)[main_q_idx],
					&hash_idx, &is_representative);
                        if (is_representative) {
                          HashmapValueT &val =
                              cst_dev_params.d_hashmap_values.lane(
                                  ilane)[hash_idx];
                          uint32_t min;
                          GetMinFromPackedArgminUInt64(
                              val.min_and_argmin_int_cost_u64, &min);
                          unsigned long long new_pack;
                          PackArgminInUInt64(min, extra_prev_tokens_offset,
                                             &new_pack);
                          val.min_and_argmin_int_cost_u64 = new_pack;
                        }
		}
	}
}

// Step4: We now know where to store our extra prev tokens in
// d_main_q_extra_prev_tokens.
// We will now move the tokens that need to be moved (when multiple tokens are
// associated to the same FST state)
// into d_main_q_extra_prev_tokens. In d_main_q_info, we will store the location
// of that list [offset,size]
// so that when backtracking, when we read d_main_q_info[token_idx], we know
// where to look to have the list
// of the same-state tokens
// It is the last step of the
// emitting_preprocess_and_list_extra_prev_tokens_step[i]_kernel pipeline
__global__ void emitting_preprocess_and_list_extra_prev_tokens_step4_kernel(
    DeviceParams cst_dev_params, KernelParams params) {
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    const LaneCounters *lane_counters =
        cst_dev_params.d_lanes_counters.lane(ilane);
    const int32 ichannel = lane_counters->channel_to_compute;
    const int main_q_end = lane_counters->main_q_narcs_and_end.y;
    // Previous frames have filled d_main_q_extra_prev_tokens.
    // d_main_q_extra_prev_tokens was then flushed to host. We want to set the
    // global
    // (global in the sense "for all frames") offset on where to read it the
    // h_all_tokens_extra_prev_tokens_ on host.
    // adding the main_q_extra_prev_tokens_global_offset for that
    const int prev_global_idx =
        lane_counters->main_q_extra_prev_tokens_global_offset;
    KALDI_CUDA_DECODER_1D_KERNEL_LOOP(main_q_idx, main_q_end) {
      // We'll take care of token at main_q_idx
      // Loading hashmap information about token.state
      bool is_representative;
      int32 hash_idx;
      GetFSTStateHashIndex(
          cst_dev_params.d_main_q_state_hash_idx.lane(ilane)[main_q_idx],
          &hash_idx, &is_representative);

      HashmapValueT val = cst_dev_params.d_hashmap_values.lane(ilane)[hash_idx];
      // How many tokens are associated with that fst state token.state
      int same_count = val.count;
      bool must_move_to_extra_prev_tokens = (same_count > 1);
      if (must_move_to_extra_prev_tokens) {
        // Moving to the extra_prev_tokens list.
        // Some of those tokens have an extra cost (compared to the best cost
        // for that FST state)
        // Generating and saving that extra cost. We will use it when generating
        // the lattice.
        CostType token_cost = orderedIntToFloat(
            cst_dev_params.d_main_q_state_and_cost.channel(ichannel)[main_q_idx]
                .y);
	uint32_t best_int_cost;
        // Where to write this state list in d_main_q_extra_prev_tokens
	uint32_t extra_prev_tokens_offset;
	unsigned long long pack = val.min_and_argmin_int_cost_u64;
	GetMinFromPackedArgminUInt64(pack, &best_int_cost);
	GetArgFromPackedArgminUInt64(pack, &extra_prev_tokens_offset);
        CostType best_cost = orderedIntToFloat((int)best_int_cost);
        CostType extra_cost = token_cost - best_cost;
	assert(!is_representative || extra_cost == 0.0f);
        // Loading the token to be moved
        InfoToken inf_tok =
            cst_dev_params.d_main_q_info.lane(ilane)[main_q_idx];
        CostType acoustic_cost =
            cst_dev_params.d_main_q_acoustic_cost.lane(ilane)[main_q_idx];
        // Place of that specific token in the extra_prev_tokens sublist of that
        // specific FST state
        int32 local_idx =
            cst_dev_params.d_main_q_n_extra_prev_tokens_local_idx.lane(
                ilane)[main_q_idx];
        // Saving the location of the extra prev tokens for that state into that
        // InfoToken
        SetSameFSTStateTokensList(
            prev_global_idx + extra_prev_tokens_offset, same_count,
            &cst_dev_params.d_main_q_info.lane(ilane)[main_q_idx]);
        // Where to write this token in d_main_q_extra_prev_tokens
        int32 list_idx = extra_prev_tokens_offset + local_idx;
        // Moving token. Also saving extra_cost
        cst_dev_params.d_main_q_extra_prev_tokens.lane(ilane)[list_idx] =
            inf_tok;
        cst_dev_params.d_main_q_extra_and_acoustic_cost.lane(
            ilane)[list_idx] = {extra_cost, acoustic_cost};
        assert(inf_tok.prev_token >= (lane_counters->main_q_global_offset -
                                      cst_dev_params.main_q_capacity) &&
               inf_tok.prev_token <=
                   (lane_counters->main_q_global_offset + main_q_end));
      }
    }
  }
}

// Clear the hashmaps after use
// Each element in the map has a representative in the main_q
// Everyone of those representatives has the responsability to reset their
// corresponding value in the hashmap
// Once this kernel returns, the hashmaps are cleared
__global__ void clear_hashmap_kernel(DeviceParams cst_dev_params,
                                     KernelParams params) {
  const int nlanes = params.nlanes_used;
  KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(ilane, nlanes) {
    LaneCounters *lane_counters = cst_dev_params.d_lanes_counters.lane(ilane);
    const int main_q_end = lane_counters->main_q_narcs_and_end.y;
    KALDI_CUDA_DECODER_1D_KERNEL_LOOP(main_q_idx, main_q_end) {
      bool is_representative;
      int32 hash_idx;
      GetFSTStateHashIndex(
          cst_dev_params.d_main_q_state_hash_idx.lane(ilane)[main_q_idx],
          &hash_idx, &is_representative);
      // Representative owns a state. Each representative resets its associated
      // token.state
      // in the hashmap
      if (is_representative) {
        cst_dev_params.d_hashmap_values.lane(ilane)[hash_idx] =
            KALDI_CUDA_DECODER_HASHMAP_NO_VAL;  // clear
      }
    }
  }
}

// Kernels wrappers

void SaveChannelsStateFromLanesKernel(const dim3 &grid, const dim3 &block,
                                      const cudaStream_t &st,
                                      const DeviceParams &cst_dev_params,
                                      const KernelParams &kernel_params) {
  save_channels_state_from_lanes_kernel<<<grid, block, 0, st>>>(cst_dev_params,
                                                                kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void LoadChannelsStateInLanesKernel(const dim3 &grid, const dim3 &block,
                                    const cudaStream_t &st,
                                    const DeviceParams &cst_dev_params,
                                    const KernelParams &kernel_params) {
  load_channels_state_in_lanes_kernel<<<grid, block, 0, st>>>(cst_dev_params,
                                                              kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void InitDecodingOnDeviceKernel(const dim3 &grid, const dim3 &block,
                                const cudaStream_t &st,
                                const DeviceParams &cst_dev_params,
                                const KernelParams &kernel_params) {
  init_decoding_on_device_kernel<<<grid, block, 0, st>>>(cst_dev_params,
                                                         kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void InitializeInitialLaneKernel(const dim3 &grid, const dim3 &block,
                                 const cudaStream_t &st,
                                 const DeviceParams &cst_dev_params) {
  initialize_initial_lane_kernel<<<grid, block, 0, st>>>(cst_dev_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void ResetForFrameAndEstimateCutoffKernel(const dim3 &grid, const dim3 &block,
                                          const cudaStream_t &st,
                                          const DeviceParams &cst_dev_params,
                                          const KernelParams &kernel_params) {
  reset_for_frame_and_estimate_cutoff_kernel<<<grid, block, 0, st>>>(
      cst_dev_params, kernel_params);
}

template <bool IS_EMITTING>
void ExpandArcsKernel(const dim3 &grid, const dim3 &block,
                      const cudaStream_t &st,
                      const DeviceParams &cst_dev_params,
                      const KernelParams &kernel_params) {
  expand_arcs_kernel<IS_EMITTING><<<grid, block, 0, st>>>(cst_dev_params,
                                                          kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

template <bool IS_EMITTING>
void PostExpandKernel(const dim3 &grid, const dim3 &block,
                      const cudaStream_t &st,
                      const DeviceParams &cst_dev_params,
                      const KernelParams &kernel_params) {
  post_expand_kernel<IS_EMITTING><<<grid, block, 0, st>>>(cst_dev_params,
                                                          kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void PostContractAndPreprocessKernel(const dim3 &grid, const dim3 &block,
                                     const cudaStream_t &st,
                                     const DeviceParams &cst_dev_params,
                                     const KernelParams &kernel_params) {
  post_contract_and_preprocess_kernel<<<grid, block, 0, st>>>(cst_dev_params,
                                                              kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void NonEmittingPreprocessAndContractKernel(const dim3 &grid, const dim3 &block,
                                            const cudaStream_t &st,
                                            const DeviceParams &cst_dev_params,
                                            const KernelParams &kernel_params) {
  nonemitting_preprocess_and_contract_kernel<<<grid, block, 0, st>>>(
      cst_dev_params, kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void FillHashmapWithMainQKernel(const dim3 &grid, const dim3 &block,
                                const cudaStream_t &st,
                                const DeviceParams &cst_dev_params,
                                const KernelParams &kernel_params) {
  fill_hashmap_with_main_q_kernel<<<grid, block, 0, st>>>(cst_dev_params,
                                                          kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void EmittingPreprocessAndListExtraPrevTokensStep1Kernel(
    const dim3 &grid, const dim3 &block, const cudaStream_t &st,
    const DeviceParams &cst_dev_params, const KernelParams &kernel_params) {
  emitting_preprocess_and_list_extra_prev_tokens_step1_kernel<<<grid, block, 0,
                                                                st>>>(
      cst_dev_params, kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void EmittingPreprocessAndListExtraPrevTokensStep2Kernel(
    const dim3 &grid, const dim3 &block, const cudaStream_t &st,
    const DeviceParams &cst_dev_params, const KernelParams &kernel_params) {
  emitting_preprocess_and_list_extra_prev_tokens_step2_kernel<<<grid, block, 0,
                                                                st>>>(
      cst_dev_params, kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void EmittingPreprocessAndListExtraPrevTokensStep3Kernel(
    const dim3 &grid, const dim3 &block, const cudaStream_t &st,
    const DeviceParams &cst_dev_params, const KernelParams &kernel_params) {
  emitting_preprocess_and_list_extra_prev_tokens_step3_kernel<<<grid, block, 0,
                                                                st>>>(
      cst_dev_params, kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void EmittingPreprocessAndListExtraPrevTokensStep4Kernel(
    const dim3 &grid, const dim3 &block, const cudaStream_t &st,
    const DeviceParams &cst_dev_params, const KernelParams &kernel_params) {
  emitting_preprocess_and_list_extra_prev_tokens_step4_kernel<<<grid, block, 0,
                                                                st>>>(
      cst_dev_params, kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void ComputeLaneOffsetsKernel(const dim3 &grid, const dim3 &block,
                              const cudaStream_t &st,
                              const DeviceParams &cst_dev_params,
                              const KernelParams &kernel_params) {
  compute_lane_offsets_kernel<<<grid, block, 0, st>>>(cst_dev_params,
                                                      kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

template <typename T>
void ConcatenateLanesDataKernel(const dim3 &grid, const dim3 &block,
                                const cudaStream_t &st,
                                const DeviceParams &cst_dev_params,
                                const KernelParams &kernel_params,
                                const LaneMatrixView<T> &src, T *concat,
                                int32 *lane_offsets) {
  concatenate_lanes_data_kernel<<<grid, block, 0, st>>>(
      cst_dev_params, kernel_params, src, concat, lane_offsets);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void InitHashmapKernel(const dim3 &grid, const dim3 &block,
                       const cudaStream_t &st,
                       const DeviceParams &cst_dev_params) {
  init_hashmap_kernel<<<grid, block, 0, st>>>(cst_dev_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void ClearHashmapKernel(const dim3 &grid, const dim3 &block,
                        const cudaStream_t &st,
                        const DeviceParams &cst_dev_params,
                        const KernelParams &kernel_params) {
  clear_hashmap_kernel<<<grid, block, 0, st>>>(cst_dev_params, kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void ComputeCostsHistogramKernel(const dim3 &grid, const dim3 &block,
                                 const cudaStream_t &st,
                                 const DeviceParams &cst_dev_params,
                                 const KernelParams &kernel_params,
                                 bool use_aux_q) {
  compute_costs_histogram_kernel<<<grid, block, 0, st>>>(
      cst_dev_params, kernel_params, use_aux_q);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void UpdateBeamUsingHistogramKernel(const dim3 &grid, const dim3 &block,
                                    const cudaStream_t &st,
                                    const DeviceParams &cst_dev_params,
                                    const KernelParams &kernel_params,
                                    bool use_aux_q) {
  update_beam_using_histogram_kernel<<<grid, block, 0, st>>>(
      cst_dev_params, kernel_params, use_aux_q);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void FinalizeProcessNonEmittingKernel(const dim3 &grid, const dim3 &block,
                                      const cudaStream_t &st,
                                      const DeviceParams &cst_dev_params,
                                      const KernelParams &kernel_params) {
  finalize_process_non_emitting_kernel<<<grid, block, 0, st>>>(cst_dev_params,
                                                               kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void GetBestCostStep1Kernel(const dim3 &grid, const dim3 &block,
                            const cudaStream_t &st,
                            const DeviceParams &cst_dev_params,
                            const KernelParams &kernel_params, bool isfinal,
                            CostType fst_zero) {
  get_best_cost_step1_kernel<<<grid, block, 0, st>>>(
      cst_dev_params, kernel_params, isfinal, fst_zero);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void GetBestCostStep2Kernel(const dim3 &grid, const dim3 &block,
                            const cudaStream_t &st,
                            const DeviceParams &cst_dev_params,
                            const KernelParams &kernel_params, bool isfinal,
                            CostType fst_zero) {
  get_best_cost_step2_kernel<<<grid, block, 0, st>>>(
      cst_dev_params, kernel_params, isfinal, fst_zero);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void GetBestCostStep3Kernel(const dim3 &grid, const dim3 &block,
                            const cudaStream_t &st,
                            const DeviceParams &cst_dev_params,
                            const KernelParams &kernel_params) {
  get_best_cost_step3_kernel<<<grid, block, 0, st>>>(cst_dev_params,
                                                     kernel_params);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

template void ExpandArcsKernel<true>(const dim3 &grid, const dim3 &block,
                                     const cudaStream_t &st,
                                     const DeviceParams &cst_dev_params,
                                     const KernelParams &params);
template void ExpandArcsKernel<false>(const dim3 &grid, const dim3 &block,
                                      const cudaStream_t &st,
                                      const DeviceParams &cst_dev_params,
                                      const KernelParams &params);
template void PostExpandKernel<true>(const dim3 &grid, const dim3 &block,
                                     const cudaStream_t &st,
                                     const DeviceParams &cst_dev_params,
                                     const KernelParams &params);
template void PostExpandKernel<false>(const dim3 &grid, const dim3 &block,
                                      const cudaStream_t &st,
                                      const DeviceParams &cst_dev_params,
                                      const KernelParams &params);

template void ConcatenateLanesDataKernel<InfoToken>(
    const dim3 &grid, const dim3 &block, const cudaStream_t &st,
    const DeviceParams &cst_dev_params, const KernelParams &params,
    const LaneMatrixView<InfoToken> &src, InfoToken *concat,
    int32 *lane_offsets);

template void ConcatenateLanesDataKernel<CostType>(
    const dim3 &grid, const dim3 &block, const cudaStream_t &st,
    const DeviceParams &cst_dev_params, const KernelParams &params,
    const LaneMatrixView<CostType> &src, CostType *concat, int32 *lane_offsets);

template void ConcatenateLanesDataKernel<float2>(
    const dim3 &grid, const dim3 &block, const cudaStream_t &st,
    const DeviceParams &cst_dev_params, const KernelParams &params,
    const LaneMatrixView<float2> &src, float2 *concat, int32 *lane_offsets);

template void ConcatenateLanesDataKernel<int32>(
    const dim3 &grid, const dim3 &block, const cudaStream_t &st,
    const DeviceParams &cst_dev_params, const KernelParams &params,
    const LaneMatrixView<int32> &src, int32 *concat, int32 *lane_offsets);

}  // end namespace cuda_decoder
}  // end namespace kaldi
