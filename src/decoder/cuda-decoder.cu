// decoder/cuda-decoder.cu

// Copyright      2018  Zhehuai Chen; Hugo Braun; Justin Luitjens

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

#include <algorithm>
#include <nvToolsExt.h>
#include <cuda_runtime_api.h>
#include <float.h>
#include <math.h>
#include <cub/cub.cuh>

#include "decoder/cuda-decoder.h"
#include "fstext/remove-eps-local.h"

namespace kaldi {

typedef CudaDecoder::StateId StateId;
typedef CudaDecoder::processTokens_params processTokens_params;
typedef CudaDecoder::Token Token;

#define COMPUTE_DEGREES_DIMX 64
#define EXPAND_ARCS_DIMX 64
#define EXPAND_ARCS_NTHDS 1e5
#define NONEM_LT_DIMX 1024
#define FILL_COSTS_DIMX 256
// Below this value, we launch the persistent kernel for NonEmitting
#define NONEM_LT_MAX_NARCS (4 * NONEM_LT_DIMX)

// device functions called by __global__ functions

// Used to reset lookup table between frames
// Using the queue to reset only the values needed
// Also takes care of resetting cutof
DEVICE static inline void _reset_lookup_kernel(processTokens_params *params,
        bool reset = true) {
    int q_offset = *params->tok_from_d_;
    int q_end = *params->tok_to_d_;

    // Avoiding a kernel call just to reset the cutoff
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // reset flag shows the last iter is emit or not
        CUDA_PRINTF(2, "5 %d %d %d %d %f %d\n", params->frame - reset, q_end - q_offset,
                    *params->active_tok_cnt_d, !reset, *params->cutoff_d_, *params->tot_narcs_d_);
        *params->active_tok_cnt_d = 0;
    }
    if (reset) {
        for (int idx = q_offset + blockIdx.x * blockDim.x + threadIdx.x;
                idx < q_end;
                idx += blockDim.x * gridDim.x) {

            StateId state = params->d_q[idx];

            params->d_lookup[state]  = pack_cost_idx_into_uint64(FLT_MAX, -1);
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) *params->cutoff_d_ = FLT_MAX;
    }
}


/*
    This kernel is responsible for :

    1) Read a token from the input queue
    2) Compute the outgoing degree of that token.next_state. For that :
       -> If that token is suboptimal (cutoff, best_cost), degree = 0
       -> Otherwise, we set degree using WFST graph

    3) Compute prefix sums of those degrees within the block :
        -> We store those "local prefix sums" in narcs_scan_d_. Another kernel will finish the job
        -> We save the sum of all degrees in that block (narcs_blksum_scan_d_)

    4) The last block alive compute the prefix sums of narcs_blksum_scan_d_.
        -> We save it, it will be needed to compute global_scan
        -> We now have the total number of arcs overall, we save it to tot_narcs_h_

    this function is always followed by _finalize_degrees_scan_kernel
*/

DEVICE static inline void _compute_degrees_kernel(processTokens_params* params) {

    typedef cub::BlockScan<int, COMPUTE_DEGREES_DIMX> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ int blk_scan_offset;
    __shared__ int is_last_CTA;

    // for degree & scan
    int queue_offset = *params->tok_from_d_;
    int queue_end = *params->tok_to_d_;
    int queue_size = queue_end - queue_offset;

    // for prune
    BaseFloat cutoff = *params->cutoff_d_;
    bool hist = params->max_active < queue_size && params->frame > 1;
    int32 hist_local[MAX_HISTOGRAM_SIZE];
    if (hist) memset(hist_local, 0, params->histogram_prev_toks.Size());

    // for all tokens before recombination in the queue
    for (int block_offset = blockDim.x * blockIdx.x;
            block_offset < queue_size;
            block_offset += gridDim.x * blockDim.x) {
        int idx = queue_offset + block_offset + threadIdx.x;
        int degree = 0;

        if (idx < queue_end) {
            StateId state_idx = params->d_q[idx];
            BaseFloat cost = params->d_q_info[idx].cost;

            if (cost < cutoff) {
                // details of packing can be referred to pack_cost_idx_into_uint64
                int ptr = unpack_idx_from_uint64(params->d_lookup[state_idx]);
                // the token alive after recombination
                if (ptr == idx) {
                    int start = params->arc_offset_d[state_idx];
                    int end = params->arc_offset_d[state_idx + 1];
                    degree = end - start; // the number of out-going arcs
                    params->arc_offset_pertok_d_[idx - queue_offset] = start;
                    // count how many tokens alive after recombination
                    if (params->active_tok_cnt_d) atomicAdd(params->active_tok_cnt_d, 1);
                    // if the token is alive, add it into histogram
                    if (hist) params->histogram_prev_toks.AddScore2LocalHist(cost, hist_local);
                }
            }
        }

        int scan;
        BlockScan(temp_storage).ExclusiveSum(degree, scan);
        if (idx < queue_end)
            params->narcs_scan_d_[idx - queue_offset] = scan;
        // the last thread keep the block scan result
        if (threadIdx.x == (COMPUTE_DEGREES_DIMX - 1)) 
            params->narcs_blksum_scan_d_[block_offset / COMPUTE_DEGREES_DIMX] =
                (scan + degree); // scan is exclusive
        
        // if there's another iteration, we'll reuse temp_storage
        if ((block_offset + gridDim.x * blockDim.x) < queue_end)
            __syncthreads();
    }
    // aggregate the per-thread histogram together; we can do it in block at first
    if (hist) params->histogram_prev_toks.AggregateLocalHist(hist_local);

    // to obtain the last CTA
    if (threadIdx.x == 0) {
        int old = atomicAdd(params->n_CTA_d_, 1);
        blk_scan_offset = 0; // will be used if last CTA, avoiding a second sync
        is_last_CTA = (old == (gridDim.x - 1));
    }
    __syncthreads(); // for is_last_CTA + temp_storage reuse if last CTA

    // The last block alive takes care of scan of block sums
    if (is_last_CTA) {
        __threadfence();
        if (threadIdx.x == 0) 
            *params->n_CTA_d_ = 0;

        // following value can be different than gridDim.x
        int total_blk_val = (queue_size + COMPUTE_DEGREES_DIMX - 1) /
                            COMPUTE_DEGREES_DIMX;

        for (int blk_idx_off = 0;
                blk_idx_off < total_blk_val;
                blk_idx_off += blockDim.x) {
            int blk_idx = blk_idx_off + threadIdx.x;

            int blk_sum = (blk_idx < total_blk_val) ? params->narcs_blksum_scan_d_[blk_idx] :
                          0;

            int blk_scan;
            BlockScan(temp_storage).ExclusiveSum(blk_sum, blk_scan);
            blk_scan += blk_scan_offset;

            if (blk_idx < total_blk_val) {
                params->narcs_blksum_scan_d_[blk_idx] = blk_scan;
            }

            __syncthreads(); // blk_scan_offset + reuse temp_storage
            if (threadIdx.x == (COMPUTE_DEGREES_DIMX - 1)) {
                int total = blk_scan + blk_sum;
                blk_scan_offset = total;
            }
        }
        __syncthreads(); // blk_scan_offset
        if (threadIdx.x == 0) {
            *params->tot_narcs_d_ = blk_scan_offset;
            *params->tot_narcs_h_ = blk_scan_offset; // pinned memory
        }
    }
}

/*
    After _compute_degrees_kernel, computes global prefix sum with block prefix sum and block offsets
    Another method is computing lower and upper bound to restrain the binary search in expand, e.g.
    We can store the following index:
    params->d_lowerbound[j + params->d_degrees_scan[idx]] = idx;
    This can removes main bottleneck of expand. However, it introduces overhead here.
    After comparison, we remain the current implementation.
    Details can be referred to binsearch_maxle .
*/
DEVICE static inline void _finalize_degrees_scan_kernel(processTokens_params *params) {
    int q_off = *params->tok_from_d_;
    int q_end = *params->tok_to_d_;
    int q_size = q_end - q_off;

    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
            idx < q_size;
            idx += blockDim.x * gridDim.x) {
        int blk_idx = idx / blockDim.x;
        int blk_scan_offset = params->narcs_blksum_scan_d_[blk_idx];
        params->narcs_scan_d_[idx] += blk_scan_offset;
    }
}

// cuda __global__ functions

// initialization of the lookup table, used before first frame
__global__
static void _init_lookup_kernel(uint64 *state_pack_d, int size) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < size;
            idx += blockDim.x * gridDim.x)
        state_pack_d[idx]  = pack_cost_idx_into_uint64(FLT_MAX, -1);
}

// a combination of degree calculation, scan and reset to reduce kernel launch time
__global__
static void _compute_degrees_with_reset_kernel(processTokens_params params,
        bool reset = true) {
    int rank0 = threadIdx.x == 0 && blockIdx.x == 0;
    _compute_degrees_kernel(&params);
    // the above scan result is needed in _finalize_degrees_scan_kernel
    grid_sync(params.barrier); 
    if (rank0 && params.frame > 1 && *params.active_tok_cnt_d > params.max_active)
        params.histogram_prev_toks.GetCutoff(params.cutoff_prev,
                                             params.max_active, 0);
    _reset_lookup_kernel(&params, reset);
    _finalize_degrees_scan_kernel(&params);
}


// find best cutoff before _expand_arcs_kernel to guarantee consistent results
__global__
static void _get_cutoff(processTokens_params params) {
    typedef cub::BlockScan<int, EXPAND_ARCS_DIMX> BlockScan;
    typedef cub::BlockReduce<BaseFloat, EXPAND_ARCS_DIMX> BlockReduce;

    __shared__ typename BlockScan::TempStorage temp_storage_scan;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

    __shared__ BaseFloat global_cutoff;

    const int total_narcs = *params.tot_narcs_d_;
    const int old_q_offset = *params.tok_from_d_;
    const int old_q_size = *params.tok_to_d_ - old_q_offset;

    if (threadIdx.x == 0) 
        global_cutoff = *params.cutoff_d_;

    for (int block_offset = blockDim.x * blockIdx.x;
            block_offset < total_narcs;
            block_offset += gridDim.x * blockDim.x) {
        int th_idx = block_offset + threadIdx.x;
        bool valid_input = (th_idx < total_narcs);

        BaseFloat total_cost = FLT_MAX;
        int arc_idx;
        StateId arc_next_state;
        int q_idx;

        if (valid_input) {
            // given a thread index, find out the arc it processes  
            // and the corresponding token, the arc comes from
            // firstly obtain the corresponding token 
            q_idx = old_q_offset + binsearch_maxle(params.narcs_scan_d_, th_idx, 0,
                                                   old_q_size - 1);
            // the outgoing arc number of the token, used to obtain arc_idx
            int lower_bound = params.narcs_scan_d_[q_idx - old_q_offset];
            // the starting arc_id of out-going arcs from the token
            int arc_offset_start = params.arc_offset_pertok_d_[q_idx - old_q_offset];
            // the arc id in the WFST
            arc_idx = arc_offset_start + (block_offset + threadIdx.x - lower_bound);

            // after deciding the arc_id, do the token passing through the arc
            arc_next_state = params.arc_nextstates[arc_idx];
            BaseFloat arc_weight = params.arc_weights[arc_idx];
            int arc_ilabel = params.is_emitting ? params.arc_ilabels[arc_idx] : 0;
            BaseFloat accoustic_cost = (arc_ilabel != 0) ?
                                       -params.cuda_decodable.LogLikelihood(
                                           arc_ilabel) : 0.0;
            BaseFloat old_tok_cost = params.d_q_info[q_idx].cost;
            total_cost = accoustic_cost + arc_weight + old_tok_cost;
            // partial token recombination to reduce the number of the following atomic_min
            BaseFloat next_state_cost = unpack_cost_from_uint64(
                                            params.d_lookup[arc_next_state]);
            if (total_cost > next_state_cost) {
                total_cost = FLT_MAX;
                valid_input = false;
            }
        }

        // do block reduce to reduce the number of the following atomic_min
        BaseFloat threacutoff_d_ = (total_cost < FLT_MAX) ? (total_cost + params.beam) :
                                   FLT_MAX;
        BaseFloat new_block_cutoff = BlockReduce(temp_storage_reduce).Reduce(
                                         threacutoff_d_, cub::Min());
        if (threadIdx.x == 0) {
            if (new_block_cutoff < global_cutoff) {
                BaseFloat new_global_cutoff = atomic_min(params.cutoff_d_, new_block_cutoff);
                new_global_cutoff = min(new_global_cutoff, new_block_cutoff);
                // update the local cutoff to reduce the number of the following atomic_min
                global_cutoff = new_global_cutoff;
            }
        }
        __syncthreads(); // for BlockReduce
    }
}


// This kernel propagates arcs from the current queue to the new queue
// The last block alive moves forward the queues indexes
// The main bottleneck is the first binary search.
// If we want to remove that bottleneck, cf comments on _finalize_degrees_scan_kernel
__global__
static void _expand_arcs_kernel(processTokens_params params) {
    int rank0 = threadIdx.x == 0 && blockIdx.x == 0;
    // reduce a kernel for histogram_prev_toks
    if (rank0) params.histogram_prev_toks.Initialize(*params.cutoff_d_ - params.beam);

    typedef cub::BlockScan<int, EXPAND_ARCS_DIMX> BlockScan;
    typedef cub::BlockReduce<BaseFloat, EXPAND_ARCS_DIMX> BlockReduce;

    __shared__ typename BlockScan::TempStorage temp_storage_scan;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;
    __shared__ int new_q_block_off;

    const int total_narcs = *params.tot_narcs_d_;
    const int old_q_offset = *params.tok_from_d_;
    const int old_q_size = *params.tok_to_d_ - old_q_offset;

    // Keeping the whole CTA alive, we'll have syncs
    for (int block_offset = blockDim.x * blockIdx.x;
            block_offset < total_narcs;
            block_offset += gridDim.x * blockDim.x) {

        int th_idx = block_offset + threadIdx.x;
        bool valid_input = (th_idx < total_narcs);

        BaseFloat total_cost = FLT_MAX;
        int arc_idx;
        StateId arc_next_state;
        int q_idx;

        BaseFloat old_tok_cost;
        if (valid_input) {
            // given a thread index, find out the arc it processes  
            // and the corresponding token, the arc comes from
            // firstly obtain the corresponding token 
            q_idx = old_q_offset + binsearch_maxle(params.narcs_scan_d_, th_idx, 0,
                                                   old_q_size - 1);
            // the outgoing arc number of the token, used to obtain arc_idx
            int lower_bound = params.narcs_scan_d_[q_idx - old_q_offset];
            // the starting arc_id of out-going arcs from the token
            int arc_offset_start = params.arc_offset_pertok_d_[q_idx - old_q_offset];
            // the arc id in the WFST
            arc_idx = arc_offset_start + (block_offset + threadIdx.x - lower_bound);

            // after deciding the arc_id, do the token passing through the arc
            arc_next_state = params.arc_nextstates[arc_idx];
            BaseFloat arc_weight = params.arc_weights[arc_idx];
            int arc_ilabel = params.is_emitting ? params.arc_ilabels[arc_idx] : 0;
            BaseFloat accoustic_cost = (arc_ilabel != 0) ?
                        -params.cuda_decodable.LogLikelihood(arc_ilabel) : 0.0;
            old_tok_cost = params.d_q_info[q_idx].cost;
            total_cost = accoustic_cost + arc_weight + old_tok_cost;
            BaseFloat next_state_cost = unpack_cost_from_uint64(
                                            params.d_lookup[arc_next_state]);
            // partial token recombination
            if (total_cost > next_state_cost) {
                total_cost = FLT_MAX;
                valid_input = false;
            }
        }

        BaseFloat cutoff = *params.cutoff_d_;
        // decide whether the token will be passed through its out-going arcs
        int has_successor = (total_cost < cutoff && valid_input
                             && old_tok_cost < *params.cutoff_prev) ? 1 : 0;
        // the following block scan and atomic operation is to decide the token 
        // index of the thread in this frame
        int new_q_idx_block;
        BlockScan(temp_storage_scan).ExclusiveSum(has_successor,
                new_q_idx_block); 
        if (threadIdx.x == (EXPAND_ARCS_DIMX - 1)) {
            int total_block = new_q_idx_block + has_successor; // exclusive sum
            new_q_block_off = atomicAdd(params.tok_end_d_, total_block);
        }
        __syncthreads(); // for newQueue_block_off + reuse temp_storage_scan + global cutoff

        // we have decided the token index
        int new_q_index = new_q_block_off + new_q_idx_block;
        // store info of this token
        if (has_successor) {
            params.d_q[new_q_index] = arc_next_state;
            Token new_tok_info;
            new_tok_info.cost = total_cost;
            new_tok_info.prev_token = q_idx;
            new_tok_info.arc_idx = arc_idx;
            params.d_q_info[new_q_index] = new_tok_info;
        }
        // do a token recombination in the lookup table
        // details of this idea can be referred to pack_cost_idx_into_uint64
        // it reduces, but not atomic (because of no return)
        if (has_successor) 
            atomicMin((unsigned long long *)&params.d_lookup[arc_next_state],
                      (unsigned long long)pack_cost_idx_into_uint64(total_cost, new_q_index));
    }

    // Last block alive moves forward the queue
    if (threadIdx.x == 0) {
        int old = atomicAdd(params.n_CTA_d_, 1);
        if (old == (gridDim.x - 1)) {
            // The last block alive takes care of preparing for next iter
            __threadfence(); // we want last value of tok_end_d_
            int final_end = *params.tok_end_d_;
            *params.tot_ntok_h_ = final_end - *params.tok_to_d_;
            *params.tok_from_d_ = *params.tok_to_d_;
            *params.tok_to_d_ = final_end;

            *params.n_CTA_d_ = 0;
            // Saving position of curr_token for this frame
            // We'll need to reset tok_from_d_ for next frame
            if (params.is_emitting) 
                *params.cur_tok_from_d_ = *params.tok_from_d_;
        }
    }

}

// Reached final kernel
__global__
static void _reached_final_kernel(StateId *d_q, const int *tok_from_d_,
                                  const int *tok_to_d_, BaseFloat *final, 
                                  float fst_zero, int *reached_final_h_) {
    int q_offset = *tok_from_d_;
    int q_end = *tok_to_d_;

    for (int idx = q_offset + blockDim.x * blockIdx.x + threadIdx.x;
            idx < q_end;
            idx += blockDim.x * gridDim.x) {

        StateId state = d_q[idx];
        float final_val = final[state];
        if (final_val != fst_zero)
            *reached_final_h_ = 1;
    }
}

// Used to find best costs.
__global__
static void _fill_costs_kernel(StateId *d_q, Token *d_q_info,
                               const int *tok_from_d_, const int *tok_to_d_,
                               uint64 *state_pack_d_, BaseFloat *d_final, bool final) {
    int q_offset = *tok_from_d_;
    int q_end = *tok_to_d_;

    for (int idx = q_offset + blockIdx.x * blockDim.x + threadIdx.x;
            idx < q_end;
            idx += blockDim.x * gridDim.x) {
        BaseFloat cost = d_q_info[idx].cost;

        if (final) {
            StateId state = d_q[idx];
            cost += d_final[state];
        }
        state_pack_d_[idx - q_offset] = pack_cost_idx_into_uint64(cost, idx);
    }

}


// one thread with multiple global memory load. But it avoids a massive memcpy D2H
// we can replace it with memory management in future
__global__
static void _get_best_path_kernel(int best_token_idx_in_all_tokens,
                                  StateId *d_all_tokens, Token
                                  *d_all_tokens_info, int *reversed_path_d_, int *path_size) {

    int tok_idx = best_token_idx_in_all_tokens;
    int idx = 0;

    while (tok_idx != INT_MIN) {
        int arc_idx = d_all_tokens_info[tok_idx].arc_idx;
        reversed_path_d_[idx++] = arc_idx;

        int old_tok_idx = tok_idx;
        tok_idx = d_all_tokens_info[tok_idx].prev_token;
        assert(old_tok_idx > tok_idx);
    }
    *path_size = idx;
}

/*
    Persistent kernel for a single CTA
    Used to avoid calling multiple kernels for the tail of non emitting
    (lots of iterations with small number of arcs)
    Code is greatly simplified because we can have only one CTA alive

    It repeats until new queue empty:
        1) Computes degrees
        2) Compute scan
        3) Expand arcs

    1 and 2 are not done on the first iteration, because it's already done
    (by corresponding kernels)

    At the end, this kernel finalize the computation for current frame,
    setting the queue to the complete current token queue
    so that it's ready for next ProcessEmitting
*/
__launch_bounds__(NONEM_LT_DIMX, 1)
__global__
static void _process_nonem_longtail(processTokens_params params) {
    typedef cub::BlockScan<int, NONEM_LT_DIMX> BlockScan;
    typedef cub::BlockReduce<float, NONEM_LT_DIMX> BlockReduce;

    __shared__ typename BlockScan::TempStorage temp_storage_scan;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

    __shared__ int total_narcs;
    __shared__ int new_q_end;

    BaseFloat cutoff;
    int old_q_offset = *params.tok_from_d_;
    int new_q_offset = *params.tok_to_d_;

    if (threadIdx.x == 0) {
        new_q_end = *params.tok_end_d_;
        total_narcs = *params.tot_narcs_d_;
    }

    __syncthreads();

    int old_q_size = new_q_offset - old_q_offset;  // move to end

    cutoff = *params.cutoff_d_;

    // we need the offsets ready for the global updates at the very end of this kernel
    new_q_offset = old_q_offset;
    bool first = true;
    int total_at = 0;
    int rank0 = threadIdx.x == 0 && blockIdx.x == 0;

    while (old_q_size > 0) {
        // Step 0 : move queues
        old_q_offset = new_q_offset;
        new_q_offset = new_q_end;

        if (!first) {

            if (threadIdx.x == 0)  {
                total_narcs = 0;
            }

            // Step 1 : compute_degrees
            for (int local_q_idx = threadIdx.x;
                    local_q_idx < old_q_size;
                    local_q_idx += blockDim.x) {

                int global_q_idx = old_q_offset + local_q_idx;

                StateId state = params.d_q[global_q_idx];
                BaseFloat cost = params.d_q_info[global_q_idx].cost;

                int degree = 0;
                if (cost < cutoff) {
                    int ptr = unpack_idx_from_uint64(params.d_lookup[state]);

                    if (ptr == global_q_idx) {
                        int start = params.arc_offset_d[state];
                        int end = params.arc_offset_d[state + 1];
                        degree = end - start;
                        params.arc_offset_pertok_d_[local_q_idx] = start;
                        if (params.active_tok_cnt_d) atomicAdd(params.active_tok_cnt_d, 1);
                    }
                }

                params.narcs_scan_d_[local_q_idx] = degree;
            }

            // Step 2 : Scan
            for (int block_off = 0;
                    block_off < old_q_size;
                    block_off += blockDim.x) {

                int local_q_idx = block_off + threadIdx.x;

                int degree = (local_q_idx < old_q_size)
                             ? params.narcs_scan_d_[local_q_idx]
                             : 0;
                int lscan;
                BlockScan(temp_storage_scan).ExclusiveSum(degree, lscan);
                int scan = lscan + total_narcs;

                if (local_q_idx < old_q_size)
                    params.narcs_scan_d_[local_q_idx] = scan;

                if (local_q_idx == 0) assert(lscan == 0);
                __syncthreads(); // total_narcs
                if (threadIdx.x == (NONEM_LT_DIMX - 1)) {
                    int total_in_block = lscan + degree;
                    total_narcs += total_in_block;
                }
            }
        } else {
            first = false;
        }

        if (rank0) {
            CUDA_PRINTF(4, "4.0 %f %d %d\n", cutoff, old_q_size, *params.active_tok_cnt_d);
            total_at += *params.active_tok_cnt_d;
            *params.active_tok_cnt_d = 0;
        }
        __syncthreads(); //total_narcs

        // Step 3 : expand arcs
        for (int block_offset = 0;
                block_offset < total_narcs;
                block_offset += blockDim.x) {

            int th_idx = block_offset + threadIdx.x;
            bool valid_input = (th_idx < total_narcs);

            BaseFloat total_cost = FLT_MAX;
            int arc_idx;
            StateId arc_next_state;
            int q_idx, local_q_idx = -1;

            BaseFloat old_tok_cost;
            if (valid_input) {
                // get from token idx
                local_q_idx = binsearch_maxle(params.narcs_scan_d_, th_idx, 0, old_q_size - 1);

                int lower_bound = params.narcs_scan_d_[local_q_idx];
                int arc_offset_start = params.arc_offset_pertok_d_[local_q_idx];
                q_idx = old_q_offset + local_q_idx;

                arc_idx = arc_offset_start + (th_idx - lower_bound);

                arc_next_state = params.arc_nextstates[arc_idx];
                BaseFloat arc_weight = params.arc_weights[arc_idx];
                BaseFloat next_state_cost = unpack_cost_from_uint64(
                                                params.d_lookup[arc_next_state]);
                old_tok_cost = params.d_q_info[q_idx].cost;

                total_cost = arc_weight + old_tok_cost;

                if (total_cost > next_state_cost) {
                    total_cost = FLT_MAX;
                    valid_input = false;
                }
            }

            BaseFloat threacutoff_d_ = (total_cost < FLT_MAX) ? (total_cost + params.beam) :
                                       FLT_MAX;
            BaseFloat new_block_cutoff = BlockReduce(temp_storage_reduce).Reduce(
                                             threacutoff_d_, cub::Min());

            int has_successor = (total_cost < cutoff && valid_input
                                 && old_tok_cost < *params.cutoff_prev) ? 1 : 0;
            int new_q_idx_block, new_q_index;
            BlockScan(temp_storage_scan).ExclusiveSum(has_successor, new_q_idx_block);
            if (has_successor) {
                new_q_index = new_q_end + new_q_idx_block;
                params.d_q[new_q_index] = arc_next_state;

                Token new_tok_info;
                new_tok_info.cost = total_cost;
                new_tok_info.prev_token = q_idx;
                new_tok_info.arc_idx = arc_idx;
                params.d_q_info[new_q_index] = new_tok_info;
            }
            if (has_successor)
                atomicMin((unsigned long long *)&params.d_lookup[arc_next_state],
                          (unsigned long long )pack_cost_idx_into_uint64(total_cost, new_q_index));

            if (threadIdx.x == (NONEM_LT_DIMX - 1)) {
                int total_in_block = new_q_idx_block + has_successor; // exclusive sum
                new_q_end += total_in_block;
            }
        }

        __syncthreads(); // new_q_end

        old_q_size = new_q_end - new_q_offset;

    }
    if (threadIdx.x == 0) {
        // Next step is ProcessEmitting of next frame, from is currToken_offset
        *params.tok_from_d_ = *params.cur_tok_from_d_;
        *params.tok_to_d_ = new_q_end;
        *params.tok_end_d_ = new_q_end;
        *params.tot_ntok_h_ = new_q_end - *params.tok_from_d_;
    }
    if (rank0) CUDA_PRINTF(3, "4 %f %d %d\n", cutoff,
                               *params.tok_to_d_ - *params.tok_from_d_, total_at);
}


// CudaLatticeDecoder Implementation

CudaDecoder::CudaDecoder(const CudaFst &fst, const TransitionModel &trans_model,
                         const CudaDecoderConfig &config): fst_(fst),
    trans_model_(trans_model),
    config_(config) {

    cudaStreamCreate(&stream_comp);
    cudaStreamCreate(&stream_ll);
    cudaEventCreateWithFlags(&event_ll, cudaEventDisableTiming);

    cudaMalloc(&cur_tok_from_d_, sizeof(int));
    cudaMalloc(&tok_from_d_, sizeof(int));
    cudaMalloc(&tok_to_d_, sizeof(int));
    cudaMalloc(&tok_end_d_, sizeof(int));

    cudaMalloc(&tot_narcs_d_, sizeof(int));
    cudaMallocHost(&tot_narcs_h_, sizeof(int));

    cudaMalloc(&token_stateid_d_, config.max_tokens * sizeof(StateId));
    cudaMalloc(&token_stateid_d_Info, config.max_tokens * sizeof(Token));

    cudaMallocHost(&tot_ntok_h_, sizeof(int));

    int max_token_frame = config_.max_tokens_per_frame;
    cudaMalloc(&narcs_scan_d_, max_token_frame * sizeof(int));
    cudaMalloc(&narcs_blksum_scan_d_,
               (max_token_frame / COMPUTE_DEGREES_DIMX + 2)* sizeof(int));
    cudaMalloc(&arc_offset_pertok_d_, max_token_frame * sizeof(int));

    cudaMalloc(&state_pack_d_, sizeof(uint64)*fst_.NumStates());

    cudaMallocHost(&reached_final_h_, sizeof(int));

    cudaMalloc(&reversed_path_d_, config_.max_len * sizeof(int));
    reversed_path_h_ = (int*)malloc(config_.max_len * sizeof(int));

    cudaMalloc(&cutoff_d_, sizeof(float));
    cudaMalloc(&cutoff_prev_d_, sizeof(float));

    cudaMalloc(&path_size_d_, sizeof(int));
    cudaMalloc(&n_CTA_d_, sizeof(int));

    cudaMalloc((void**)&active_tok_cnt_d_, 1 * sizeof(int32));
    cudaMalloc((void**)&barrier_d_, 1 * sizeof(int32));
    cudaMemset(active_tok_cnt_d_, 0, sizeof(int));
    cudaMemset(barrier_d_, 0, sizeof(int));

    const std::vector<int32>& id2pdf_id = trans_model_.GetId2pdf();
    int data_size = id2pdf_id.size() * sizeof(int);
    cudaMalloc((void**)&id2pdf_d_, data_size);
    cudaMemcpy(id2pdf_d_, id2pdf_id.data(), data_size, cudaMemcpyHostToDevice);
    histogram_prev_toks_.Allocate(config_.beam,
                                  (int32)(config_.beam * 0.5), 1.0);

    CU_SAFE_CALL(cudaGetLastError());
}

CudaDecoder::~CudaDecoder() {
    cudaStreamDestroy(stream_comp);
    cudaStreamDestroy(stream_ll);
    cudaEventDestroy(event_ll);

    cudaFree(cur_tok_from_d_);
    cudaFree(tok_from_d_);
    cudaFree(tok_to_d_);
    cudaFree(tok_end_d_);

    cudaFree(tot_narcs_d_);
    cudaFreeHost(tot_narcs_h_);

    cudaFree(token_stateid_d_);
    cudaFree(token_stateid_d_Info);

    cudaFreeHost(tot_ntok_h_);

    cudaFree(narcs_scan_d_);
    cudaFree(narcs_blksum_scan_d_);
    cudaFree(arc_offset_pertok_d_);
    cudaFree(state_pack_d_);
    cudaFreeHost(reached_final_h_);
    cudaFree(reversed_path_d_);
    free(reversed_path_h_);
    cudaFree(cutoff_d_);
    cudaFree(cutoff_prev_d_);
    cudaFree(path_size_d_);
    cudaFree(n_CTA_d_);

    cudaFree(active_tok_cnt_d_);
    cudaFree(barrier_d_);

    cudaFree(id2pdf_d_);

    histogram_prev_toks_.Free();
}

void CudaDecoder::InitDecoding() {
    InitLookup();

    StateId start_state = fst_.Start();
    KALDI_ASSERT(start_state != fst::kNoStateId);

    Token it_init;
    it_init.cost = StdWeight::One().Value();
    it_init.prev_token = INT_MIN;
    it_init.arc_idx = -1;

    cudaMemcpy(token_stateid_d_, &start_state, sizeof(StateId),
               cudaMemcpyHostToDevice);
    cudaMemcpy(token_stateid_d_Info, &it_init, sizeof(Token), cudaMemcpyHostToDevice);

    cudaMemset(cur_tok_from_d_, 0, sizeof(int));
    cudaMemset(tok_from_d_, 0, sizeof(int));

    // Init state
    int one = 1;
    cudaMemcpy(tok_to_d_, &one, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tok_end_d_, &one, sizeof(int), cudaMemcpyHostToDevice);
    *tot_ntok_h_ = 1;
    uint64 packv = pack_cost_idx_into_uint64(it_init.cost, 0);
    cudaMemcpy(&state_pack_d_[start_state], &packv, sizeof(uint64),
               cudaMemcpyHostToDevice);

    float cutoff = FLT_MAX;
    cudaMemcpy(cutoff_d_, &cutoff, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cutoff_prev_d_, &cutoff, sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(n_CTA_d_, 0, sizeof(int));

    cudaMemset(active_tok_cnt_d_, 0, sizeof(int));
    cudaMemset(barrier_d_, 0, sizeof(int));

    CU_SAFE_CALL(cudaGetLastError());
    num_frames_decoded_ = 0;

    ProcessNonemitting();
}

void CudaDecoder::Decode(MatrixChunker *decodable) {
    InitDecoding();
    while ( !decodable->IsLastFrame(num_frames_decoded_ - 1)) {
        bool last_frame = decodable->IsLastFrame(num_frames_decoded_ - 0);

        PUSH_RANGE("ComputeLogLikelihoods", 3);
        CuMatrix<BaseFloat> *post_chunk;
        decodable->LogLikelihoodChunk(num_frames_decoded_, &post_chunk, stream_ll);
        cudaEventRecord(event_ll, stream_ll);
        POP_RANGE;
        DecodeChunk(post_chunk);

        if (last_frame) {
            KALDI_VLOG(5) << "last frame: " << NumFramesDecoded();
            break;
        }
    }
}

void CudaDecoder::DecodeChunk(CuMatrix<BaseFloat> *post_chunk) {
    int chunk_used_len = 0;
    while (chunk_used_len < post_chunk->NumRows()) {
        num_frames_decoded_++;
        // one frame from the chunk
        cuda_decodable_ = CuMatrixScaledMapper(id2pdf_d_, config_.acoustic_scale, 
                                               post_chunk->Row(chunk_used_len).Data());

        cudaStreamWaitEvent(stream_comp, event_ll, 0);

        ProcessEmitting();

        ProcessNonemitting();

        chunk_used_len++;
    }
}


bool CudaDecoder::ReachedFinal() const {
    dim3 grid, block;
    block.x = 256;
    assert(*tot_ntok_h_);
    grid.x = DIV_ROUND_UP(*tot_ntok_h_, block.x);

    _reached_final_kernel <<< grid, block>>>(token_stateid_d_, tok_from_d_, tok_to_d_,
            fst_.final_d, StdWeight::Zero().Value(), reached_final_h_);
    cudaDeviceSynchronize();
    return *reached_final_h_;
}

// Outputs an FST corresponding to the single best path through the lattice.
bool CudaDecoder::GetBestPath(Lattice *fst_out, bool use_final_probs) const {
    nvtxRangePushA("GetBestPath");
    BaseFloat best_cost_final;
    int arg_best_final;
    bool isfinal = ReachedFinal();
    GetBestCost(&best_cost_final, &arg_best_final, isfinal);

    int h_curr_token_offset;
    cudaMemcpy(&h_curr_token_offset, tok_from_d_, sizeof(int),
               cudaMemcpyDeviceToHost);

    int h_best_token_idx = arg_best_final;
    h_best_token_idx += h_curr_token_offset;

    cudaMemset(path_size_d_, 0, sizeof(int));

    _get_best_path_kernel <<< 1, 1>>>(h_best_token_idx, token_stateid_d_,
                                      token_stateid_d_Info, reversed_path_d_, path_size_d_);

    cudaDeviceSynchronize();

    int h_path_size;
    cudaMemcpy(&h_path_size, path_size_d_, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(reversed_path_h_, reversed_path_d_, h_path_size * sizeof(int),
               cudaMemcpyDeviceToHost);


    fst_out->DeleteStates();

    // We can assert first state equals to root
    StateId cur_state = fst_out->AddState();
    fst_out->SetStart(cur_state);

    // -1 for 0-indexing, -1 for ignoring starting arc
    for (int i = h_path_size - 1 - 1; i >= 1; i--) {
        int arc_idx = reversed_path_h_[i];
        LatticeArc arc(fst_.arc_ilabels_h[arc_idx], fst_.arc_olabels_h[arc_idx],
                       LatticeWeight(fst_.arc_weights_h[arc_idx], 0), 
                       fst_.arc_nextstates_h[arc_idx]);

        arc.nextstate = fst_out->AddState();
        fst_out->AddArc(cur_state, arc);
        cur_state = arc.nextstate;
    }

    if (isfinal && use_final_probs)
        fst_out->SetFinal(cur_state, LatticeWeight(
                fst_.Final(fst_.arc_nextstates_h[reversed_path_h_[0]]), 0.0));
    else
        fst_out->SetFinal(cur_state, LatticeWeight::One());

    fst::RemoveEpsLocal(fst_out);

    nvtxRangePop();
    return true;
}

BaseFloat CudaDecoder::FinalRelativeCost() const {
    if (*tot_ntok_h_ == 0)
        return FLT_MAX;

    BaseFloat best_cost;
    int arg_best;
    GetBestCost(&best_cost, &arg_best, false);

    BaseFloat best_cost_final;
    int arg_best_final;
    GetBestCost(&best_cost_final, &arg_best_final, true);

    return (best_cost_final - best_cost);
}

// TODO: Change 4-space indents to 2-space indents.

void CudaDecoder::InitLookup() {
  int nstates = fst_.NumStates();


    dim3 grid, block;
    block.x = 256;
    grid.x = DIV_ROUND_UP(nstates, block.x);

    _init_lookup_kernel <<< grid, block>>>(state_pack_d_, nstates);
}


bool CudaDecoder::ProcessToken(bool is_emitting) {
    processTokens_params params;
    InitParams(&params, is_emitting);
    ContractAndPreprocess(params);

    bool done = false;
    if (!params.is_emitting) {
        NonEmittingLongTail(params);
        CU_SAFE_CALL(cudaGetLastError());
        done = true;
    } else {
        ExpandArcs(EXPAND_ARCS_NTHDS, params);
    }

    CU_SAFE_CALL(cudaGetLastError());
    return done;
}


void CudaDecoder::ProcessEmitting() {
    nvtxRangePushA("ProcessEmitting");

    // Using emitting arc offsets
    ProcessToken(true);
    CU_SAFE_CALL(cudaGetLastError());

    nvtxRangePop();
}

void CudaDecoder::ProcessNonemitting() {
    nvtxRangePushA("ProcessNonemitting");

    // While not done, call it
    while (!ProcessToken(false));

    CU_SAFE_CALL(cudaGetLastError());
    nvtxRangePop();
}


// The distinction between emitting / non emitting depends on the argument
// passed as "arc_offset_d"
void CudaDecoder::InitParams(processTokens_params* params, bool is_emitting) {
    params->d_q = token_stateid_d_;
    params->d_q_info = token_stateid_d_Info;
    params->tok_from_d_ = tok_from_d_;
    params->tok_to_d_ = tok_to_d_;
    params->tok_end_d_ = tok_end_d_;
    params->narcs_scan_d_ = narcs_scan_d_;
    params->arc_offset_pertok_d_ = arc_offset_pertok_d_;
    params->arc_ilabels = fst_.arc_ilabels_d;
    params->tot_narcs_d_ = tot_narcs_d_;
    params->tot_narcs_h_ = tot_narcs_h_;
    params->arc_weights = fst_.arc_weights_d;
    params->arc_nextstates = fst_.arc_nextstates_d;
    params->cutoff_d_ = cutoff_d_;
    params->cutoff_prev = cutoff_prev_d_;
    params->max_active = config_.max_active;
    params->beam = config_.beam;
    params->d_lookup = state_pack_d_;
    params->is_emitting = is_emitting;
    params->cur_tok_from_d_ = cur_tok_from_d_;
    params->tot_ntok_h_ = tot_ntok_h_;
    params->n_CTA_d_ = n_CTA_d_;
    params->active_tok_cnt_d = active_tok_cnt_d_;
    params->barrier = barrier_d_;
    params->frame = num_frames_decoded_;
    params->arc_offset_d = is_emitting? fst_.e_offsets_d: fst_.ne_offsets_d;
    params->narcs_blksum_scan_d_ = narcs_blksum_scan_d_;
    params->cuda_decodable = cuda_decodable_;
    params->histogram_prev_toks = histogram_prev_toks_;
}


void CudaDecoder::ContractAndPreprocess(const processTokens_params &params) {
    dim3 grid, block;
    block.x = COMPUTE_DEGREES_DIMX;
    assert(*tot_ntok_h_);
    grid.x = DIV_ROUND_UP(*tot_ntok_h_, block.x);

    _compute_degrees_with_reset_kernel <<< grid, block, 0, stream_comp>>>(params,
            params.is_emitting);
    CU_SAFE_CALL(cudaGetLastError());
}


void CudaDecoder::ExpandArcs(int nthreads, const processTokens_params &params) {
    dim3 grid, block;
    block.x = EXPAND_ARCS_DIMX;
    grid.x = DIV_ROUND_UP(nthreads, block.x);

    _get_cutoff <<< grid, block, 0, stream_comp>>>(params);
    _expand_arcs_kernel <<< grid, block, 0, stream_comp>>>(params);
}


void CudaDecoder::NonEmittingLongTail(const processTokens_params &params) {

    dim3 grid, block;
    block.x = NONEM_LT_DIMX;
    grid.x = 1; // it is designed for the long tail
    _process_nonem_longtail <<< grid, block, 0, stream_comp>>>(params);
}

void CudaDecoder::GetBestCost(BaseFloat *min, int *arg, bool isfinal) const {
    dim3 grid, block;
    block.x = FILL_COSTS_DIMX;
    grid.x = DIV_ROUND_UP(*tot_ntok_h_, block.x);

    // using lookup as float buffer for now
    _fill_costs_kernel <<< grid, block>>>(token_stateid_d_, token_stateid_d_Info,
                                          tok_from_d_, tok_to_d_, state_pack_d_, fst_.final_d, isfinal);

    cub::KeyValuePair<int, uint64> *d_argmin;
    cudaMalloc(&d_argmin, sizeof(cub::KeyValuePair<int, int>));

    void *d_temp_storage_amin = NULL;
    size_t temp_storage_amin_bytes = 0;

    cub::DeviceReduce::ArgMin(d_temp_storage_amin, temp_storage_amin_bytes,
                              state_pack_d_, d_argmin, *tot_ntok_h_);
    cudaMalloc(&d_temp_storage_amin, temp_storage_amin_bytes);

    cub::DeviceReduce::ArgMin(d_temp_storage_amin, temp_storage_amin_bytes,
                              state_pack_d_, d_argmin, *tot_ntok_h_);

    cub::KeyValuePair<int, uint64> h_argmin;
    cudaMemcpy(&h_argmin, d_argmin, sizeof(cub::KeyValuePair<int, int>),
               cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage_amin);
    cudaFree(d_argmin);

    *min = unpack_cost_from_uint64(h_argmin.value);
    *arg = h_argmin.key;
}

} // end namespace kaldi.
