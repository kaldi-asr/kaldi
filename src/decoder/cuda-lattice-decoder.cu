// decoder/cuda-lattice-decoder.cu

// Copyright      2018  Zhehuai Chen

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http:// www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <cub/cub.cuh>

#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"

#include "cuda-decoder-utils.h"
#include "cuda-lattice-decoder.h"

namespace kaldi {

typedef CudaLatticeDecoder::Token Token;
typedef CudaLatticeDecoder::StateId StateId;
typedef CudaLatticeDecoder::TokenState TokenState;
typedef CudaLatticeDecoder::CostType CostType;
typedef CudaLatticeDecoder::TokenLookupElem TokenLookupElem;
typedef CudaLatticeDecoder::LatLink LatLink;
typedef CudaLatticeDecoder::LatLinkCompact LatLinkCompact;
typedef CudaLatticeDecoder::LatLinkVector LatLinkVector;
typedef CudaLatticeDecoder::TokenMergeVector TokenMergeVector;
typedef CudaLatticeDecoder::processTokens_params processTokens_params;
typedef CudaLatticeDecoder::LatticeProcessor LatticeProcessor;
#define CudaVector CudaLatticeDecoder::CudaVector
#define CudaMergeVector CudaLatticeDecoder::CudaMergeVector
#define PROC_DIMX 256
#define COMPUTE_DEGREES_DIMX 256
// instantiation of templates
template HOST DEVICE LatLink& CudaVector<LatLink>::operator[](uint32 idx);
template HOST DEVICE TokenState& CudaVector<TokenState>::operator[](uint32 idx);
template HOST DEVICE uint32  CudaVector<TokenState>::Size() const;
template HOST DEVICE uint32  CudaVector<LatLink>::Size() const;
template<> DEVICE void CudaMergeVector<TokenState>::StoreDataByPackIdx(
  void* temp_data_buf, int* temp_data_buf_update, int32 buf_size,
  LatticeProcessor *lattice_processor);

// inline functions

// swap code in device, as we need to instantiate it, so we define it here
template<typename T>
inline DEVICE void cuda_swap(T &a, T &b) {
  T c = a;
  a = b;
  b = c;
}

// device functions called by __global__ functions

// during token passing, re-initialize lookuptables of visited states
// to prepare for the next lookup
DEVICE static inline void _initialize_visited_states(
  TokenLookupElem *current_tokens_lookup,
  TokenMergeVector cur_toks) {
  int32 size = cur_toks.Size();
  for (int32 i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    StateId state = cur_toks[i].state; // cur_toks will be clear in PreProcessTokens
    current_tokens_lookup[state].tokenstate_idx = LOOKUP_DEACTIVE;
    // do not need to clear Token* cur_tok, as data will be re-stored in it
    // in StoreDataByPackIdx()
  }
}

// initialize cutoff used in beam pruning
DEVICE static inline void _initialize_cutoff(CostType *cutoff) { *cutoff = 1e2; }

// _find_or_add_token_arc either locates a token in TokenLookupElem
// or if necessary add a token by activating it in TokenLookupElem
// for the current frame.  The function also adds a lattice arc into the vector
// it's a GPU version of FindOrAddToken() and ForwardLink()
DEVICE static inline void _find_or_add_token_arc(processTokens_params* params,
    StateId nextstate, CostType total_cost, CostType acoustic_cost,
    TokenState* ts, uint32 j, bool add_arc, TokenState** next_ts,
    uint64 **token_pack, int32* update_idx, bool is_emit) {
  TokenLookupElem& lookup_elem = params->current_tokens_lookup[nextstate];
  // check if token is active or not.  if not, activate it
  if (lookup_elem.tokenstate_idx == LOOKUP_DEACTIVE
      && atomicCAS((int32 *)&lookup_elem.tokenstate_idx,
                   (int32)LOOKUP_DEACTIVE, (int32)LOOKUP_READY_PUSH) ==
      (int32)LOOKUP_DEACTIVE) {
    // grab sentinal to see who gets to add to cur_toks list
    // if haven't seen this token, add into hash by activating it
    // push back the TokenState, and also record its index in lookup table
    int32 tokenstate_idx = params->cur_toks.PushBack(TokenState(nextstate));
    // firstly store it, so that other threads can continue processing
    lookup_elem.tokenstate_idx = tokenstate_idx;
    int32 tok_idx_allocated =
      params->lattice_processor.GetTokenAllocIdx(tokenstate_idx);
    // do not need to clear Token* cur_tok, as data will be stored in it
    // in function StoreDataByPackIdx()
    // use tokenstate_idx + front_d (accumulate number of tokens before
    // this frame) as tok_idx_allocated
    params->cur_toks[tokenstate_idx].tok_idx_allocated = tok_idx_allocated;
    // thus we do not need to ensure tok_idx_allocated has been allocated,
    // as it will be used after this iteration and at StoreDataByPackIdx()
  }
  // need both 2 steps below, to ensure tokenstate_idx recorded correctly
  while (lookup_elem.tokenstate_idx == LOOKUP_DEACTIVE ||  // hasn't pushed
         lookup_elem.tokenstate_idx == LOOKUP_READY_PUSH);  // during pushing
  __threadfence();

  *next_ts = &params->cur_toks[lookup_elem.tokenstate_idx]; // get it using index
  if (add_arc) { // we add lattice arc except in _add_initial_token()
    Token *prev_tok = params->lattice_processor.GetTokenByExactIdx(
                        ts->tok_idx_allocated);
    int32 prev_tok_frame = (!is_emit) ? params->frame : params->frame - 1;
    int32 ts_id = (!is_emit) ?
                  params->cur_toks.GetIdxFromAddr(ts) : // process non-emit tokens
                  params->prev_toks.GetIdxFromAddr(ts); // process emit tokens
    LatLinkCompact arc(ts_id, prev_tok_frame,
                       lookup_elem.tokenstate_idx, params->frame,
                       acoustic_cost, j);
    // use pushBack (idx - start index of this frame)
    // as update item index because it is unique in each frame obtained from
    // atomicAdd in pushBack function (lat_arcs_sub_vec
    // is recorded accumulatively and cleared only at end of decoding).
    //
    // another choice is doing block scan & atomic before and get arc_idx,
    // fed it to _find_or_add_token_arc here;
    // however block scan costs 5% more time, while atomic here doesn't result
    // in explicit burdens.
    int32 ret = params->lat_arcs_sub_vec.PushBack(arc) - *params->num_arcs_till_last;
    assert(ret < params->max_lat_arc_per_frame);
    *update_idx = ret;
  }
  // get token_pack variable address for atomic based token recombination
  *token_pack = &((*next_ts)->token_pack);
  return;
}

DEVICE static inline void _find_prev_cutoff_by_histogram(processTokens_params* params) {
  // to reduce a grid sync, we initialize it in _process_tokens()
  // params->histogram_prev_toks.Initialize(*params->cutoff - params->beam);
  bool rank0 = blockIdx.x == 0 && threadIdx.x == 0;
  int32 hist_local[MAX_HISTOGRAM_SIZE];
  memset(hist_local, 0, params->histogram_prev_toks.Size());

  int32 size = params->prev_toks.Size();
  for (int32 i = threadIdx.x + blockIdx.x * blockDim.x;
       i < size; i += gridDim.x * blockDim.x) {
    TokenState ts = params->prev_toks[i];
    Token * tok = params->lattice_processor.GetTokenByExactIdx(ts.tok_idx_allocated);
    params->histogram_prev_toks.AddScore2LocalHist(tok->cost_, hist_local);
  }
  params->histogram_prev_toks.AggregateLocalHist(hist_local);

  grid_sync(params->barrier);

  if (rank0)
    params->histogram_prev_toks.GetCutoff(params->cutoff_prev,
                                          params->max_active, params->verbose);
}

DEVICE static inline void _find_best_cutoff(processTokens_params * params) {
  int32 size = params->prev_toks.Size();
  // frame 0 don't have params->cutoff
  if (size > params->max_active && params->frame > 1) 
    _find_prev_cutoff_by_histogram(params);

  typedef cub::BlockReduce<BaseFloat, COMPUTE_DEGREES_DIMX> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

  CostType local_cutoff = INFINITY;
  bool is_emitting = true;
  TokenMergeVector &tok_vec = is_emitting ? params->prev_toks : params->cur_toks;
  const int total_narcs = *params->d_q_token_from_narcs;
  const int old_q_offset = 0;
  const int old_q_size = tok_vec.Size();

  // as we'll have syncs, use block_offset < total_narcs to keep the whole CTA alive
  for (int block_offset = blockDim.x * blockIdx.x; block_offset < total_narcs;
       block_offset += gridDim.x * blockDim.x) {
    int th_idx = block_offset + threadIdx.x;
    bool valid_input = (th_idx < total_narcs);

    BaseFloat total_cost = FLT_MAX, acoustic_cost, weight;
    int arc_idx, q_idx;
    TokenState* ts;
    Token *tok = NULL;

    if (valid_input) {
      // q_idx of the previous token
      // we do not keep q_idx but obtain it on-the-fly, details in binsearch_maxle()
      q_idx = old_q_offset + binsearch_maxle(params->d_degrees_scan, th_idx, 0,
                                             old_q_size - 1);
      // arc number lower bound of this previous token
      int arc_number_offset = params->d_degrees_scan[q_idx - old_q_offset];
      ts  = &tok_vec[q_idx];
      tok = params->lattice_processor.GetTokenByExactIdx(ts->tok_idx_allocated);
      // arc id lower bound of this previous token
      int arc_id_offset = params->d_q_arc_offset[q_idx - old_q_offset];
      arc_idx = arc_id_offset + (block_offset + threadIdx.x - arc_number_offset);
      int arc_ilabel = is_emitting ? params->arc_ilabels[arc_idx] : 0;
      acoustic_cost = (arc_ilabel != 0) ? -params->cuda_decodable.LogLikelihood(
                       arc_ilabel) : 0.0;
      weight = params->arc_weights[arc_idx];
      total_cost = tok->cost_ + weight + acoustic_cost + params->beam;
      if (total_cost < local_cutoff)
        local_cutoff = total_cost;
    }
  } // for block

  // reduce inside block first
  BaseFloat new_block_cutoff = BlockReduce(temp_storage_reduce).Reduce(local_cutoff,
                               cub::Min());
  if (threadIdx.x == 0 && new_block_cutoff != INFINITY) {
    atomic_min(params->cutoff, new_block_cutoff);
  }
}

DEVICE static inline void _compute_degrees_scan(processTokens_params * params,
                                                bool is_emitting) {
  typedef cub::BlockScan<int, COMPUTE_DEGREES_DIMX> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ int blk_scan_offset;
  __shared__ int is_last_CTA;

  TokenMergeVector &tok_vec = is_emitting ? params->prev_toks : params->cur_toks;
  const uint32 *d_offsets = is_emitting ? params->e_offsets : params->ne_offsets;
  int queue_offset = 0;
  int queue_end = tok_vec.Size();
  int queue_size = queue_end - queue_offset;
  // as we'll have syncs, use block_offset < total_narcs to keep the whole CTA alive
  for (int block_offset = blockDim.x * blockIdx.x;
       block_offset < queue_size;
       block_offset += gridDim.x * blockDim.x) {
    int idx = queue_offset + block_offset + threadIdx.x;
    int degree = 0;
    if (idx < queue_end) { // compute degrees
      TokenState& ts = tok_vec[idx];
      StateId state_idx = ts.state;
      if (is_emitting || tok_vec.IsUpdated(idx)) {
        int start = d_offsets[state_idx];
        int end = d_offsets[state_idx + 1];
        degree = end - start;
        params->d_q_arc_offset[idx - queue_offset] = start;
      }
    }
    int scan;
    BlockScan(temp_storage).ExclusiveSum(degree, scan); // do scanning
    if (idx < queue_end)
      params->d_degrees_scan[idx - queue_offset] = scan;
    if (threadIdx.x == (COMPUTE_DEGREES_DIMX - 1)) {
      params->d_block_sums[block_offset / COMPUTE_DEGREES_DIMX] =
        (scan + degree); //  scan is exclusive
    }

    // if there's another iteration, we'll reuse temp_storage
    if ((block_offset + gridDim.x * blockDim.x) < queue_end) __syncthreads();
  }

  // for is_last_CTA (the last block to finish)
  if (threadIdx.x == 0) {
    int old = atomicAdd(params->d_n_CTA_done, 1);
    blk_scan_offset = 0; // will be used if last CTA, avoiding a second sync
    is_last_CTA = (old == (gridDim.x - 1));
  }
  __syncthreads(); // is_last_CTA + temp_storage reuse if last CTA
  if (is_last_CTA) { // The last block alive takes care of scan of block sums
    __threadfence();
    if (threadIdx.x == 0) *params->d_n_CTA_done = 0;

    // following value can be different than gridDim.x
    int total_blk_val = (queue_size + COMPUTE_DEGREES_DIMX - 1) /
                        COMPUTE_DEGREES_DIMX;
    for (int blk_idx_off = 0; // obtain block sum
         blk_idx_off < total_blk_val;
         blk_idx_off += blockDim.x) {
      int blk_idx = blk_idx_off + threadIdx.x;
      int blk_sum = (blk_idx < total_blk_val) ? params->d_block_sums[blk_idx] : 0;
      int blk_scan;
      BlockScan(temp_storage).ExclusiveSum(blk_sum, blk_scan);
      blk_scan += blk_scan_offset;
      if (blk_idx < total_blk_val) {
        params->d_block_sums_scan[blk_idx] = blk_scan;
      }
      __syncthreads(); // blk_scan_offset + reuse temp_storage
      if (threadIdx.x == (COMPUTE_DEGREES_DIMX - 1)) {
        int total = blk_scan + blk_sum;
        blk_scan_offset = total;
      }
    }
    __syncthreads(); // blk_scan_offset
    if (threadIdx.x == 0) {
      *params->d_q_token_from_narcs = blk_scan_offset;
    }
  }

  grid_sync(params->barrier); // after finishing above scanning

  // add block sum to scan result
  int q_off = queue_offset;
  int q_end = queue_end;
  int q_size = q_end - q_off;
  for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
       idx < q_size;
       idx += blockDim.x * gridDim.x) {
    int blk_idx = idx / blockDim.x;
    int blk_scan_offset = params->d_block_sums_scan[blk_idx]; 
    params->d_degrees_scan[idx] += blk_scan_offset;
  }
}

DEVICE static inline void _process_emit_or_nemit_tokens(processTokens_params *params,
    bool is_emitting = true, volatile int32 *modified = NULL, int isize = 0) {
  CostType cutoff = *params->cutoff;
  TokenMergeVector &tok_vec = is_emitting ? params->prev_toks : params->cur_toks;
  // same vec to be modified in the following, so we have to use in-parameter
  int32 size = is_emitting ? tok_vec.Size() : isize; 
  const int total_narcs = *params->d_q_token_from_narcs;
  const int old_q_offset = 0;
  const int old_q_size = tok_vec.Size();

  // Keeping the whole CTA alive, we'll have syncs
  for (int block_offset = blockDim.x * blockIdx.x;
       block_offset < total_narcs;
       block_offset += gridDim.x * blockDim.x) {
    int th_idx = block_offset + threadIdx.x;
    bool valid_input = (th_idx < total_narcs);

    StateId nextstate;
    BaseFloat total_cost = FLT_MAX, acoustic_cost, weight;
    int arc_idx, q_idx;
    TokenState* ts;
    Token *tok = NULL;

    if (valid_input) {
      // q_idx of the previous token
      // we do not keep q_idx but obtain it on-the-fly, details in binsearch_maxle()
      q_idx = old_q_offset + binsearch_maxle(params->d_degrees_scan, th_idx, 0,
                                             old_q_size - 1);

      // arc number lower bound of this previous token
      int arc_number_offset = params->d_degrees_scan[q_idx - old_q_offset];
      ts  = &tok_vec[q_idx];
      tok = params->lattice_processor.GetTokenByExactIdx(ts->tok_idx_allocated);
      assert(tok);
      BaseFloat old_tok_cost = tok->cost_;
      // arc id lower bound of this previous token
      int arc_id_offset = params->d_q_arc_offset[q_idx - old_q_offset];
      arc_idx = arc_id_offset + (block_offset + threadIdx.x - arc_number_offset);
      int arc_ilabel = is_emitting ? params->arc_ilabels[arc_idx] : 0;
      nextstate = params->arc_nextstates[arc_idx];
      weight = params->arc_weights[arc_idx];
      acoustic_cost = (arc_ilabel != 0) ? -params->cuda_decodable.LogLikelihood(
                       arc_ilabel) : 0.0;
      total_cost = acoustic_cost + weight + old_tok_cost;

      if (tok->cost_ > *params->cutoff_prev || total_cost > cutoff) continue;

      // not prune out
      uint64* token_pack;
      TokenState *next_ts = NULL;
      int32 update_idx;
      // get cur_tok&token_pack addr
      _find_or_add_token_arc(params, nextstate, total_cost,
                             acoustic_cost, ts, arc_idx, true, &next_ts, &token_pack,
                             &update_idx, is_emitting);
      // 1st stage of 2-pass atomic token recombination
      // get cur_te&new_token_pack here
      // details in the definition of pack_cost_idx_into_uint64()
      uint64 new_token_pack = pack_cost_idx_into_uint64(-total_cost, update_idx);
      uint64 ret = atomicMax((unsigned long long *)token_pack,
                             (unsigned long long)new_token_pack);
      if (ret < new_token_pack) {
        assert(update_idx < params->max_lat_arc_per_frame);
        Token* cur_te = params->token_per_arc + update_idx;
        fast_store8(cur_te, &(Token(acoustic_cost + weight, tok)));
        params->token_per_arc_update[update_idx] = 1;
        if (!is_emitting) (*modified) = true; // show that we need another iteration
      }
    } // end valid_input
  } // for block

  grid_sync(params->barrier); // after finishing all tokens
  // 2nd stage of 2-pass atomic token recombination
  params->cur_toks.StoreDataByPackIdx(params->token_per_arc,
                                      params->token_per_arc_update, params->numArcs,
                                      &(params->lattice_processor));
}

// end of inline device functions

// cuda __global__ functions

// before token passing, initialize lookuptables of all states
// to prepare for the lookup
__global__
static void _initialize_all_states(TokenLookupElem *current_tokens_lookup,
                                   int32 numStates, int32 *barrier) {
  for (int32 i = blockIdx.x * blockDim.x + threadIdx.x; i < numStates;
       i += blockDim.x * gridDim.x) {
    current_tokens_lookup[i].tokenstate_idx = LOOKUP_DEACTIVE;
    // do not need to clear Token* cur_tok, as data will be re-stored in it
    // in StoreDataByPackIdx()
  }
  grid_sync(barrier);
  // we do not allocate token, so do not need this
  // if (blockIdx.x == 0 && threadIdx.x == 0) allocator.AdvanceFront(numStates);
}

// initialize by add the first token in the start state of WFST
__global__
static void _add_initial_token(processTokens_params params, StateId state) {
  TokenState *next_ts = NULL;
  uint64* token_pack;
  int32 j = 0;
  int32 update_idx = 0;
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  // putting this here to avoid extra kernel launch cost
  _initialize_cutoff(params.cutoff);
  _initialize_cutoff(params.cutoff_prev);

  _find_or_add_token_arc(&params, state, 0, // add first token
                         0, NULL, j, false,  &next_ts,
                         &token_pack, &update_idx, false);
  uint64 new_token_pack = pack_cost_idx_into_uint64(0, update_idx);
  Token* cur_te = params.token_per_arc + update_idx;
  params.token_per_arc_update[update_idx] = 1;
  fast_store8(cur_te, &(Token(0, NULL)));
  atomicMax((unsigned long long *)token_pack, (unsigned long long)new_token_pack);
  params.cur_toks.StoreDataByPackIdx(params.token_per_arc,
                                     params.token_per_arc_update, params.numArcs,
                                     &(params.lattice_processor));
}

// providing additional information of (maxThreadsPerBlock, minBlocksPerMultiprocessor)
// to the compiler to make more threads and blocks reside on GPU
__launch_bounds__(PROC_DIMX, 64)
__global__
static void _process_tokens(processTokens_params params, bool is_init = false) {
  bool rank0 = blockIdx.x == 0 && threadIdx.x == 0;

  if (!is_init) { 
    if (rank0) params.cur_toks.Clear(0); // clear here to reduce a kernel call

    _compute_degrees_scan(&params, true); //for emit
    grid_sync(params.barrier); // after finishing all tokens

    _find_best_cutoff(&params);
    grid_sync(params.barrier);

    // to reduce a grid sync, we initialize here for next frame
    if (rank0)
      params.histogram_prev_toks.Initialize(*params.cutoff - params.beam);
  } else if (rank0) {
    *params.num_arcs_till_last = 0;
  }

  volatile int32 *modified0 = params.modified;
  // modified flag for next/last iteration
  volatile int32 *modified1 = params.modified + 1;
  *modified1 = false;
  CostType cutoff = *params.cutoff;

  if (rank0) {
    *modified0 = false;
    *modified1 = false;
  }
  if (!is_init) { 
    _process_emit_or_nemit_tokens(&params);
    grid_sync(params.barrier);  // ensure cur_toks size is final
  }

  // debug
  int32 tok_E, tok_E_sc;
  int32 itv = params.verbose > 2 ? 1 : 10;
  // if without -G, this code will be optimized out
  if (rank0 && params.verbose > 0 && params.frame % itv == 0) {
    tok_E = params.cur_toks.Size();
    tok_E_sc = *params.d_q_token_from_narcs;
  }
  int32 cnt = 0;
  uint32 size = 0;
  do {
    cnt++;
    _compute_degrees_scan(&params, false); //for non-emit
    // wait for everyone to read size and modified0
    size = params.cur_toks.Size();
    // swap buffers: double buffered to avoid extra sync when resetting
    // modified to false, 3% speedup
    // if we use more modified, we can reduce more grid sync,
    // but will make the program complexer
    cuda_swap(modified0, modified1);
    if (rank0) *modified1 = false;
    grid_sync(params.barrier); // for _compute_degrees_scan()
    _process_emit_or_nemit_tokens(&params, false, modified0, size);

    // wait for everyone to finish process tokens and writes modified0
    // we have sync in the end of _process_emit_or_nemit_tokens
    // thus we can reduce a grid_sync(params.barrier);
  } while ((*modified0) == true && cnt < 10);
  if (rank0 && params.verbose > 2 && params.frame % itv == 0) {
    if (is_init) {
      CUDA_PRINTF(1, "TK: %i %i %i %f\n", params.frame, tok_E,
                params.cur_toks.Size(), cutoff);
    } else {
      CUDA_PRINTF(1, "TK: %i %i %i %i %i %f %f\n", params.frame, tok_E,
                params.cur_toks.Size(), tok_E_sc, *params.d_q_token_from_narcs,
                cutoff, params.cuda_decodable.LogLikelihood(1));
    }
  }

  // process lattice before allocate new toks to TokenState
  params.lattice_processor.CollectToksPerFrame(params.cur_toks, params.frame);
  // accumulatively store lattice arcs
  params.lattice_processor.CollectArcsPerFrame(params.lat_arcs_sub_vec,
      params.frame);
  if (rank0) {
    *params.num_arcs_till_last = params.lat_arcs_sub_vec.Size();
  }

  grid_sync(params.barrier); // after process lattice

  _initialize_visited_states(params.current_tokens_lookup, params.cur_toks);

  if (rank0) {
    // prepare for next iteration
    *params.cutoff = INFINITY;
  }
}

// providing additional information of (maxThreadsPerBlock, minBlocksPerMultiprocessor)
// to the compiler to make more threads and blocks reside on GPU
__launch_bounds__(PROC_DIMX, 64)
__global__
static void _prune_active_tokens(processTokens_params params) {
  params.lattice_processor.PruneActiveTokens(params.frame, params.lattice_beam, params.verbose);
}

// end of cuda __global__ functions

// CudaVector Implementation
template<typename T>
void CudaVector<T>::Allocate(uint32 max_size,
                             uint32* count_h, uint32* count_d, T* mem_h, T* mem_d) {
  alloc_size = 0;
  this->max_size = max_size;

  if (count_h) this->count_h = count_h;
  else cudaMallocHost(&this->count_h, sizeof(uint32));
  if (count_d) this->count_d = count_d;
  else {
    alloc_size += sizeof(uint32);
    cudaMalloc(&this->count_d, sizeof(uint32));
  }
  if (mem_h) this->mem_h = mem_h;
  else cudaMallocHost(&this->mem_h, max_size * sizeof(T));
  if (mem_d) this->mem_d = mem_d;
  else {
    alloc_size += max_size * sizeof(T);
    cudaMalloc(&this->mem_d, max_size * sizeof(T));
  }

  cudaMemset(this->count_d, 0, sizeof(uint32));
  *this->count_h = 0;
}

template<typename T>
void CudaVector<T>::Free(bool create_outside) {
  cudaFreeHost(mem_h);
  if (!create_outside) {
    cudaFree(mem_d);
  }
  cudaFreeHost(count_h);
  cudaFree(count_d);
}

template<typename T>
HOST DEVICE T& CudaVector<T>::operator[](uint32 idx) {
#ifdef __CUDA_ARCH__
  assert(idx < *count_d);
  return mem_d[idx];
#else
  assert(idx < *count_h);
  return mem_h[idx];
#endif
}

template<typename T>
HOST DEVICE const T& CudaVector<T>::operator[](uint32 idx) const {
#ifdef __CUDA_ARCH__
  assert(idx < *count_d);
  return mem_d[idx];
#else
  assert(idx < *count_h);
  return mem_h[idx];
#endif
}

// This will cause page faults back and forth when we switch from host to device.
// need to call e.g. CopySizeToHost() before this function
template<typename T>
HOST DEVICE uint32 CudaVector<T>::Size() const {
#ifdef __CUDA_ARCH__
  return *count_d;
#else
  return *count_h;
#endif
}

// push back function implemented
// by an atomic operation, where the memory is pre-allocated
template<typename T>
HOST DEVICE uint32 CudaVector<T>::PushBack(const T &val) {
#ifdef __CUDA_ARCH__
  uint32 idx = atomicAdd(count_d, 1);
#ifdef __DEBUG__
  assert(*count_d < max_size);
#else
  if (*count_d >= max_size) *count_d = max_size - 1;
#endif
  if (sizeof(T) == 16) fast_store16(mem_d + idx, &val);
  else mem_d[idx] = val;
#else
  assert(*count_h < max_size);
  uint32 idx = (*count_h)++;
  mem_h[idx] = val;
#endif
  return idx;
}

template<typename T>
HOST DEVICE void CudaVector<T>::Clear(cudaStream_t stream) {
#ifdef __CUDA_ARCH__
  *count_d = 0;
#else
  *count_h = 0;
  cudaMemsetAsync(count_d, 0, sizeof(int32), stream);
#endif
}

template<typename T>
void CudaVector<T>::Swap(CudaVector<T> &v) {
  std::swap(mem_h, v.mem_h);
  std::swap(mem_d, v.mem_d);
  std::swap(count_h, v.count_h);
  std::swap(count_d, v.count_d);
  std::swap(max_size, v.max_size);
}

// given an allocated address in vector memory, calculate its index in the vector
template<typename T>
HOST DEVICE int32 CudaVector<T>::GetIdxFromAddr(T* addr) {
#ifdef __CUDA_ARCH__
  int32 ret = addr - mem_d;
  assert(ret < *count_d && ret >= 0);
  return ret;
#else
  int32 ret = addr - mem_h;
  assert(ret < *count_h && ret >= 0);
  return ret;
#endif
}

// a series of data transfer functions between host and device
template<typename T>
void CudaVector<T>::CopyAllToHost(cudaStream_t stream) {
  cudaStreamSynchronize(stream);
  cudaMemcpy(count_h, count_d, sizeof(int32), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(mem_h, mem_d, *count_h * sizeof(T), cudaMemcpyDeviceToHost,
                  stream);
}

template<typename T>
void CudaVector<T>::CopyAllToDevice(cudaStream_t stream) {
  cudaStreamSynchronize(stream);
  cudaMemcpyAsync(count_d, count_h, sizeof(int32), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(mem_d, mem_h, *count_h * sizeof(T), cudaMemcpyHostToDevice,
                  stream);
}

template<typename T>
void CudaVector<T>::CopySizeToHost(cudaStream_t stream) {
  cudaMemcpyAsync(count_h, count_d, sizeof(int32), cudaMemcpyDeviceToHost, stream);
}

template<typename T>
void CudaVector<T>::CopySizeToDevice(cudaStream_t stream) {
  cudaMemcpyAsync(count_d, count_h, sizeof(int32), cudaMemcpyHostToDevice, stream);
}

template<typename T>
void CudaVector<T>::CopyDataToHost(cudaStream_t stream, T* to_buf,
                                   bool copy_size) {
  if (!to_buf) {
    to_buf = mem_h;
  }
  if (copy_size) cudaMemcpy(count_h, count_d, sizeof(int32),
                              cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(to_buf, mem_d, *count_h * sizeof(T), cudaMemcpyDeviceToHost,
                  stream);
}

template<typename T>
void CudaVector<T>::CopyDataToDevice(cudaStream_t stream) {
  cudaMemcpyAsync(mem_d, mem_h, *count_h * sizeof(T), cudaMemcpyHostToDevice,
                  stream);
}

// CudaVector Implementation

template<typename T>
void CudaMergeVector<T>::Allocate(uint32 max_size) {
  CudaVector<T>::Allocate(max_size);

  cudaMalloc(&mem_update_d, sizeof(int32) * max_size);
  cudaMalloc(&barrier_, sizeof(int32) * 1);

  cudaMemset(mem_update_d, 0, sizeof(int32) * max_size);
}

template<typename T>
void CudaMergeVector<T>::Free() {
  CudaVector<T>::Free();

  cudaFree(mem_update_d);
  cudaFree(barrier_);
}

template<typename T>
void CudaMergeVector<T>::Swap(CudaMergeVector<T> &v) {
  CudaVector<T>::Swap(v);
  std::swap(mem_update_d, v.mem_update_d);
}

template<typename T>
size_t CudaMergeVector<T>::GetCudaMallocBytes() {
  return CudaVector<T>::GetCudaMallocBytes() +
         sizeof(uint32) * (1 + 2 * (2)) + max_size * (sizeof(T) +
             sizeof(uint64*) + sizeof(int32));
}


template<typename T>
DEVICE void CudaMergeVector<T>::StoreDataByPackIdx(
  void* temp_data_buf, int* temp_data_buf_update, int32 buf_size,
  LatticeProcessor* lattice_processor) {
  assert(0);  // haven't implemented
}

// according to the unpack index, copy data from external buf to the inside
// buf; it's used in the 2nd stage of 2-pass atomic token recombination
// Namely, in each frame, we save the token
// information in an array whose size is the number of arcs. This
// ensures there are no write conflicts between threads since each
// arc can be accessed at most once in each frame. After passing
// all tokens, we aggregate survived packed tokens, unpack them
// to get arc indexes, and store token information from the former
// array to token data structures exploiting thread parallelism.
template<>
DEVICE void CudaMergeVector<TokenState>::StoreDataByPackIdx(
  void* temp_data_buf, int* temp_data_buf_update, int32 buf_size,
  LatticeProcessor* lattice_processor) {
  int32 tid = threadIdx.x + blockIdx.x * blockDim.x;
  int32 batch = blockDim.x * gridDim.x;
  int32 size = *count_d; // count_d is cleared in Clear() called by InitDecoding()

  for (; tid < size; tid += batch) { // thread parallelism
    uint64* pack_v = &mem_d[tid].token_pack;
    int32 idx = unpack_idx_from_uint64(*pack_v);
    assert(idx < buf_size);
    mem_update_d[(tid + 0)] = temp_data_buf_update[idx];
    if (temp_data_buf_update[idx]) temp_data_buf_update[idx] = 0;
    else continue; // if it isn't updated, just skip storing
    TokenState* to_ts = mem_d + (tid + 0);
    Token* cur_tok = ((Token *)temp_data_buf) + idx;
    Token* to_tok = lattice_processor->GetTokenByExactIdx(to_ts->tok_idx_allocated);
    fast_store8(to_tok, cur_tok); // memcpy(to_tok,cur_tok,sizeof(T));
  }
}

// check whether the item in index i of the vector is updated in this frame
// call this function after StoreDataByPackIdx()
template<typename T>
DEVICE int32 CudaMergeVector<T>::IsUpdated(int32 i) {
  if (i >= *count_d) return 0;
  return mem_update_d[i];
}

// push back function implemented
// by an atomic operation, where the memory is pre-allocated
template<typename T>
DEVICE uint32 CudaMergeVector<T>::PushBack(const T &val,
    uint64 *val_pack) {
  assert(0); //this func is deprecated
  uint32 idx = atomicAdd(count_d, 1);
  assert(*count_d < max_size);
  assert(sizeof(val) == 16); // use faster storing
  fast_store16(&mem_d[idx], &val);
  // store the pack_data pointer in 1st stage
  // mem_pack_buf_d[idx] = val_pack; // used in StoreDataByPackIdx() in 2nd stage, it's always stored in mem_d now
  return idx;
}

// LatticeProcessor Implementation
// Initialize in InitDecoding()
void LatticeProcessor::Initialize() {
  cudaMemset(arcs_apr_fr_size_d, 0, sizeof(int32) * (max_len + 2));
  cudaMemset(arcs_apr_used_d, 0, sizeof(int32));
  cudaMemset(arcs_bpr_used_d, 0, sizeof(int32));
  cudaMemset(toks_bpr_fr_sidx_d, 0, sizeof(int32) * (max_len + 2));
  cudaMemset(arcs_bpr_fr_sidx_d, 0, sizeof(int32) * (max_len + 2));
  cudaMemset(toks_num_used, 0, sizeof(int32));
}

// the return value including the cudaMallocManaged size
int32 LatticeProcessor::Allocate(int32 max_tokens_per_frame,
                                 int32 max_lat_arc_per_frame, int32 max_len,
                                 int32 max_toks, int32 max_arcs,
                                 const CudaFst& fst) {
  int32 sz;
  int32 bytes_cuda_malloc = 0;

  // before pruning
  // to reduce memory usage, we use cudaMallocManaged, which doesn't
  // allocate in GPU at once
  sz = sizeof(Token) * max_toks;
  CudaMallocManagedPreferredDevice((void**)&toks_bpr_d, sz);
  bytes_cuda_malloc += sz;
  // if we directly use managed memory from toks_bpr_d, the RTF is 10% larger
  cudaMallocHost((void**)&toks_bpr_h, sz);
  toks_buf_before_pr_size = sz / sizeof(Token);

  // to reduce memory usage, we use cudaMallocManaged, which doesn't
  // allocate in GPU at once
  sz = sizeof(LatLinkCompact) * max_arcs;
  CudaMallocManagedPreferredDevice((void**)&arcs_bpr_d, sz);
  bytes_cuda_malloc += sz;

  arcs_buf_before_pr_size = max_arcs;
  sz = sizeof(int32) * (max_len + 2);
  cudaMalloc((void**)&toks_bpr_fr_sidx_d, sz); bytes_cuda_malloc += sz;
  cudaMallocHost((void**)&toks_bpr_fr_sidx_h, sz);
  sz = sizeof(int32);
  cudaMalloc((void**)&toks_num_used, sz); bytes_cuda_malloc += sz;
  sz = sizeof(int32) * (max_len + 2);
  cudaMalloc((void**)&arcs_bpr_fr_sidx_d, sz); bytes_cuda_malloc += sz;

  // after pruning
  sz = sizeof(int32) * (max_len + 2);
  cudaMalloc((void**)&arcs_apr_fr_size_d, sz); bytes_cuda_malloc += sz;
  cudaMallocHost((void**)&arcs_apr_fr_size_h, sz);
  sz = ESTIMATED_PRUNE_RATIO * sizeof(LatLink) * max_arcs;
  // to reduce memory usage, we use cudaMallocManaged, which doesn't
  // allocate in GPU at once
  CudaMallocManagedPreferredDevice((void**)&arcs_apr_d, sz);
  bytes_cuda_malloc += sz;
  cudaMallocHost((void**)&arcs_apr_h, sz);
  sz = sizeof(int32);
  cudaMalloc((void**)&arcs_apr_used_d, sz); bytes_cuda_malloc += sz;
  cudaMalloc((void**)&arcs_bpr_used_d, sz); bytes_cuda_malloc += sz;
  cudaMallocHost((void**)&arcs_apr_used_h, sz);

  // GPU global memory temp variables
  sz = sizeof(int32);
  cudaMalloc((void**)&barrier_, sz); bytes_cuda_malloc += sz;
  sz = sizeof(int32) * 3;
  cudaMalloc((void**)&modified_d, sz); bytes_cuda_malloc += sz;
  sz = sizeof(int32) * (2);
  cudaMalloc((void**)&count_vec_acc_d, sz); bytes_cuda_malloc += sz;
  this->max_len = max_len;

  arc_ilabels = fst.arc_ilabels_d;
  arc_olabels = fst.arc_olabels_d;
  arc_weights = fst.arc_weights_d;
  return bytes_cuda_malloc;
}
void LatticeProcessor::Free() {
  // before pruning
  cudaFree(arcs_bpr_used_d);
  cudaFreeHost(arcs_apr_used_h);
  //cudaFree(toks_bpr_d);
  cudaFreeHost(toks_bpr_h);
  cudaFree(arcs_bpr_d);
  cudaFree(toks_bpr_fr_sidx_d);
  cudaFreeHost(toks_bpr_fr_sidx_h);
  cudaFree(arcs_bpr_fr_sidx_d);
  cudaFree(toks_num_used);

  // after pruning
  cudaFree(arcs_apr_fr_size_d);
  cudaFreeHost(arcs_apr_fr_size_h);
  cudaFree(arcs_apr_d);
  cudaFree(arcs_apr_used_d);

  // GPU global memory temp variables
  cudaFree(count_vec_acc_d);
  cudaFree(barrier_);
  cudaFree(modified_d);
  cudaFreeHost(arcs_apr_h);
}

DEVICE Token* LatticeProcessor::GetTokenByExactIdx(uint32 offset) {
  int32 idx = offset;
#ifdef __DEBUG__
  assert(idx >= 0 && idx < toks_buf_before_pr_size);
#else
  if (idx >= toks_buf_before_pr_size) idx = toks_buf_before_pr_size - 1;
#endif
  return toks_bpr_d + idx;
}

DEVICE int32 LatticeProcessor::GetTokenAllocIdx(uint32 offset) {
  int32 idx = *toks_num_used + offset;
#ifdef __DEBUG__
  assert(idx >= 0 && idx < toks_buf_before_pr_size);
#else
  if (idx >= toks_buf_before_pr_size) idx = toks_buf_before_pr_size - 1;
#endif
  return idx;
}

DEVICE int32 LatticeProcessor::GetTokenIdxFromAddr(Token* tok) {
  int32 ret = tok - toks_bpr_d;
  assert(ret < toks_buf_before_pr_size && ret >= 0);
  return ret;
}

// entry of lattice pruning until this frame
DEVICE void LatticeProcessor::PruneActiveTokens(int32 frame,
    BaseFloat lattice_beam, int32 verbose) {
  int32 rank0 = threadIdx.x == 0 && blockIdx.x == 0 ? 1 : 0;
  if (frame == 0) return;
  if (rank0) *arcs_apr_used_d = 0; // clear buffer index
  grid_sync(barrier_);
  for (int32 f = frame; f > 0; f--) { // prune each frame in serial
    PruneLatticeForFrame(f, 1, lattice_beam, verbose);
  }
  // by ESTIMATED_PRUNE_RATIO to reduce memory allocation and D2H data transfer
  assert(*arcs_apr_used_d < arcs_buf_before_pr_size * ESTIMATED_PRUNE_RATIO);
  if (verbose > 2 && rank0)
    CUDA_PRINTF(1, "PRt: %i %i\n", arcs_bpr_fr_sidx_d[frame + 1],
                *arcs_apr_used_d);
  *arcs_apr_used_h = *arcs_apr_used_d; // pinned memory 
}

// collect after each token passing, we store Token data in the sequence of
// TokenState vector, using continuous memory
DEVICE void LatticeProcessor::CollectToksPerFrame(
  TokenMergeVector& cur_toks_vec, int32 frame) {
  int32 tid = threadIdx.x + blockIdx.x * blockDim.x;
  int32 size = cur_toks_vec.Size();
  if (tid == 0) {
    // Set start index in the buffer of the next frame
    SetNextSidx(toks_bpr_fr_sidx_d, size, frame);
    *toks_num_used += size;
    assert(*toks_bpr_fr_sidx_d < toks_buf_before_pr_size);
  }
}

// collect after each token passing, mainly to update arcs_bpr_fr_sidx_d here
DEVICE void LatticeProcessor::CollectArcsPerFrame(LatLinkVector&
    cur_arc_array,
    int32 frame) {
  int32 rank0 = blockIdx.x == 0 && threadIdx.x == 0 ? 1 : 0;

  int32 size = cur_arc_array.Size() - *arcs_bpr_used_d; // size of current frame
  grid_sync(barrier_);
  if (rank0) {
    SetNextSidx(arcs_bpr_fr_sidx_d, size, frame);
    *arcs_bpr_used_d = cur_arc_array.Size();
    // we didn't clear cur_arc_array.count_d until the end of decoding
  }
  /*
  // we share the memory between vector&pruner, so dont need to copy between them
  for(; idx < size; idx += batch) {
    LatLink* to_arc=GetActiveArc(frame,(idx));
    fast_store32(to_arc, cur_arc_array.mem_d+idx);
    // for debug purpose
    GetActiveToken((cur_arc_array.mem_d+idx)->p1,true,frame);
    GetActiveToken(to_arc->p1,true,frame);
  }
  */
}

// AddArc function implemented
// by an atomic operation, where the memory is pre-allocated
DEVICE int32 LatticeProcessor::AddArc(LatLink* arc) {
  int32 i = atomicAdd(arcs_apr_used_d, 1);
  assert(i < arcs_buf_before_pr_size * ESTIMATED_PRUNE_RATIO);
  fast_store32(arcs_apr_d + i, arc);
}
DEVICE int32 LatticeProcessor::AddArc(LatLinkCompact* arc, int32 frame) {
  int32 i = atomicAdd(arcs_apr_used_d, 1);
  assert(i < arcs_buf_before_pr_size * ESTIMATED_PRUNE_RATIO);
  int32 frame_tok = arc->IsEmitArc() ? frame - 1 : frame;
  int32 j = arc->arc_id;
  LatLink apr_arc(arc->GetPrevTokId(), frame_tok, arc->next_tok_id, frame,
                  arc_ilabels[j], arc_olabels[j], arc_weights[j], arc->acoustic_cost);
  fast_store32(arcs_apr_d + i, &apr_arc);
}


// Set start index in the buffer of the next frame
DEVICE void LatticeProcessor::SetNextSidx(int* sidx_buf, int32 size,
    int32 frame) {
  assert(frame >= 0);
  int32 cur_sidx = sidx_buf[(frame)];
  sidx_buf[(frame + 1)] = cur_sidx + size;
}

// Get the active token indexed by a uint64 pair (frame, idx), stored in void* p
// the details of the pair can be referred to LatLink::LatLink()
DEVICE Token* LatticeProcessor::GetActiveToken(void* p, bool check,
    int32 iframe) const {
  int32 frame, id;
  DECODE_TOK_IDX_PAIR(frame, id, (uint64)p);
  if (check) assert(frame == iframe || frame == iframe - 1);
  return GetActiveToken(frame, id, check);
}

// Get the active token indexed by a uint64 pair (frame, idx)
// the details of the pair can be referred to LatLink::LatLink()
DEVICE Token* LatticeProcessor::GetActiveToken(int32 frame, int32 id_pack,
    bool check) const {

  int32 cur_sidx = toks_bpr_fr_sidx_d[frame];
  int32 id = id_pack & ((1 << 31) - 1);
  assert(cur_sidx + id < toks_buf_before_pr_size);
  Token* tok = toks_bpr_d + cur_sidx + id;
  /*
  if (check) {
    assert(tok->frame == frame);
  }
  */
  return tok;
}

// Get the active token indexed by a uint64 pair (frame, idx)
// the details of the pair can be referred to LatLink::LatLink()
DEVICE Token* LatticeProcessor::GetActiveTokenByExactId(int32 frame,
    int32 id_exact, bool check) const {
  Token* tok = toks_bpr_d + id_exact;

  if (check) {
    if (id_exact < toks_bpr_fr_sidx_d[frame]) CUDA_PRINTF(1, "h %i %i\n", id_exact,
          toks_bpr_fr_sidx_d[frame]);
    if (id_exact >= toks_bpr_fr_sidx_d[frame + 1]) CUDA_PRINTF(1, "t %i %i\n", id_exact,
          toks_bpr_fr_sidx_d[frame + 1]);
    assert(toks_bpr_fr_sidx_d[frame] <= id_exact &&
           id_exact < toks_bpr_fr_sidx_d[frame + 1]);
  }

  return tok;
}

// Get the active arc indexed by a uint64 pair (frame, idx)
// the vector memory and the start index of each frame are kept in LatticeProcessor
DEVICE LatLinkCompact* LatticeProcessor::GetActiveArc(int32 frame,
    int32 id) const {
  int32 cur_sidx = arcs_bpr_fr_sidx_d[(frame)];
  assert(cur_sidx + id < arcs_buf_before_pr_size);
  LatLinkCompact* arc = arcs_bpr_d + cur_sidx + id;
  return arc;
}

// Size of items in the frame, it is obtained from an accumulate number array
DEVICE int32 LatticeProcessor::GetSize(int* acc_len, int32 frame) const {
  int32 size = acc_len[(frame) + 1] - acc_len[(frame)];
  assert(size >= 0 && size <= arcs_buf_before_pr_size);
  return size;
}

// used in PruneLatticeForFrame()
DEVICE void LatticeProcessor::UpdateModifiedFlags(
  volatile int32 **modified0, volatile int32 **modified1,
  volatile int32 **modified2, int cnt, int32 *modified_d) {
  *modified0 = modified_d + cnt % 3;
  *modified1 = modified_d + (cnt + 1) % 3;
  *modified2 = modified_d + (cnt + 2) % 3;
}

// The parallel lattice pruning is based on the algorithm in
// LatticeFasterDecoder::PruneActiveTokens
// with necessary modifications for GPU parallelization:
// i) parallelize the iterative updating of nodes and arcs over GPU
// threads; ii) use a global arc vector to replace the linked lists in
// the old implementation, for its lack of random access features to
// enable parallel access; iii) implement the extra cost updating as
// an atomic operation to eliminate write conflicts among threads.
// When a lattice arc is pruned, we do not physically remove
// the arc, as memory allocation is expensive. Instead, we do a
// final merging step to aggregate all remaining arcs using thread
// parallelism
// We do not prune lattice nodes because: i) we need a static mapping
// for each arc to trace the previous and the next nodes before
// and after D2H memory copy. We use frame index t and vector
// index i to trace a node, thus node positions in the vector cannot
// be changed. ii) the lattice is constructed in CPU by iterating
// remaining arcs, thus nodes are implicitly pruned. iii) node D2H
// copy is done in each frame asynchronously, which does not introduce overheads.
DEVICE void LatticeProcessor::PruneLatticeForFrame(int32 frame,
    bool merge, BaseFloat lattice_beam, int32 verbose) {
  int32 prev_cidx;
  int32 c = 0;
  int32 rank0 = threadIdx.x == 0 && blockIdx.x == 0 ? 1 : 0;
  volatile int32 *modified0;
  volatile int32 *modified1;
  volatile int32 *modified2;
  int32 cnt = 0;
  UpdateModifiedFlags(&modified0, &modified1, &modified2, cnt, modified_d);
  if (rank0 && verbose > 3) CUDA_PRINTF(2, "%i %i\n", c++, GetSize(toks_bpr_fr_sidx_d,
                                          frame - 1)); // size before pruning
  {
    // initialize
    int32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    int32 size = GetSize(toks_bpr_fr_sidx_d, frame - 1);
    for (; tid < size; tid += gridDim.x * blockDim.x) {
      Token* tok = GetActiveToken(frame - 1, tid, true);
      tok->extra_cost = FLT_MAX;
    }
    if (rank0) {
      *modified0 = 1;
      *modified1 = 0;
      *modified2 = 0;
      prev_cidx = *arcs_apr_used_d;
    }
    // wait for i) last iteration(frame+1) finish ii) finish initialization
    grid_sync(barrier_);
  }

  // iteratively updates extra costs of nodes and arcs until they stop changing,
  while (cnt++ < 10 && *modified0 != 0) {
    // triple buffer to eliminate a grid sync after *modified1 = 0;
    UpdateModifiedFlags(&modified0, &modified1, &modified2, cnt, modified_d);
    // till now, threads are using modified0 & modified2, so we clear
    // *modified1 here as it won't be used before grid sync in the very below
    if (rank0) *modified1 = 0;
    // wait for every thread to enter while, which slow down by 2% here
    //grid_sync(barrier_);

    int32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    int32 size = GetSize(arcs_bpr_fr_sidx_d, frame);
    for (; tid < size; tid += gridDim.x * blockDim.x) {
      LatLinkCompact* link = GetActiveArc(frame, tid);
      int32 frame_tok = link->IsEmitArc() ? frame - 1 : frame;
      Token* next_tok = GetActiveToken(frame, link->next_tok_id, true);
      Token* tok = GetActiveToken(frame_tok, link->GetPrevTokId(), true);
      // extra cost is defined as the difference between the best
      // cost including the current arc and the best overall path.
      BaseFloat link_extra_cost = next_tok->extra_cost +
                                  ((tok->cost_ + link->acoustic_cost + 
                                  arc_weights[link->arc_id]) - next_tok->cost_);
      if (!isnan(link_extra_cost) && link_extra_cost <= lattice_beam) {
        // not prune out
        if (link_extra_cost < -1) {// debug
          CUDA_PRINTF(2, "%i %f %f %f %f %f\n", frame, next_tok->extra_cost, tok->cost_,
                      link->acoustic_cost, arc_weights[link->arc_id], next_tok->cost_);
          link_extra_cost = lattice_beam / 2;
        }
        if (link_extra_cost < tok->extra_cost) {
          atomic_min(&tok->extra_cost, link_extra_cost);
          if (*modified0 == 0) atomicAdd((int32 *)modified0, 1);
        }
      }
    }
    grid_sync(barrier_);
    if (rank0 && verbose > 3) CUDA_PRINTF(2, "%i %i\n", c++, cnt);
  }

  // final aggregate remaining arcs
  {
    int32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    int32 size = GetSize(arcs_bpr_fr_sidx_d, frame);
    for (; tid < size; tid += gridDim.x * blockDim.x) {
      LatLinkCompact* link = GetActiveArc(frame, tid);
      int32 frame_tok = link->IsEmitArc() ? frame - 1 : frame;
      Token* next_tok = GetActiveToken(frame, link->next_tok_id, true);
      Token* tok = GetActiveToken(frame_tok, link->GetPrevTokId(), true);
      BaseFloat link_extra_cost = next_tok->extra_cost +
                                  ((tok->cost_ + link->acoustic_cost + 
                                  arc_weights[link->arc_id]) - next_tok->cost_);
      if (!isnan(link_extra_cost) && link_extra_cost <= lattice_beam) {
        // not pruned out
        if (merge) {
          AddArc(link, frame);
          // link->acoustic_cost=CUDART_NAN_F;
          // don't need to delete it in original lattice
        }
      }
    }
    grid_sync(barrier_);
  }

  /*
  { // we do not prune lattice node
    // update tok
    int32 tid=threadIdx.x+blockIdx.x*blockDim.x;
    int32 size=GetSize(toks_bpr_fr_sidx_d,frame);
    for (;tid<size;tid+=gridDim.x*blockDim.x) {
      Token* tok=GetActiveToken(frame-1,tid);
      if (tok->extra_cost==FLT_MAX)
        tok->tot_cost=CUDART_NAN_F; // prune
    }
  }
  */

  // get size
  if (merge && rank0) {
    int& size_arc_of_frame = arcs_apr_fr_size_d[frame];
    size_arc_of_frame = *arcs_apr_used_d - prev_cidx;
    if (verbose > 3 || (size_arc_of_frame == 0
                        && frame != 0)) CUDA_PRINTF(1, "PR %i %i %i\n", frame,
                              GetSize(arcs_bpr_fr_sidx_d, frame), size_arc_of_frame);
  }
  // grid_sync(barrier_);
}

// copy accumulated arcs after lattice pruning till the given frame
// after obtaining the copy size, copy the buffer asynchronously
void LatticeProcessor::CopyArcsToHost(int32 frame, cudaStream_t st) {
  int32 sz;
  // one possibility is we can copy static length
  // by assuming ESTIMATED_PRUNE_RATIO parts are remained
  // sz=sizeof(LatLink)*(arcs_buf_before_pr_size*ESTIMATED_PRUNE_RATIO);

  sz = sizeof(LatLink) * (*arcs_apr_used_h); // use exact count
  cudaMemcpyAsync(arcs_apr_h, arcs_apr_d,
                  sz, cudaMemcpyDeviceToHost, st);
  sz = sizeof(int32) * (frame + 1) * (1);
  cudaMemcpyAsync(arcs_apr_fr_size_h, arcs_apr_fr_size_d,
                  sz, cudaMemcpyDeviceToHost, st);
  // clear arcs_apr_used_d in GPU during next call of pruning
}

// copy accumulated toks till the given frame
// after obtaining the copy size, copy the buffer asynchronously
void LatticeProcessor::CopyToksToHost(int32 frame, cudaStream_t st) {
  int32 sz;
  // include frame 0 count and the total count in the last element
  assert(frame <= max_len); // the max size of toks_bpr_fr_sidx_h
  sz = sizeof(int32) * (frame + 1 + 1) * (1);
  cudaMemcpy(toks_bpr_fr_sidx_h, toks_bpr_fr_sidx_d,
             sz, cudaMemcpyDeviceToHost);
  sz = sizeof(Token) * (toks_bpr_fr_sidx_h[frame + 1]);
  assert(sz); // assume we have obtain the total count
  cudaMemcpyAsync(toks_bpr_h, toks_bpr_d,
                  sz, cudaMemcpyDeviceToHost, st);
}

// get back the host data address which can be used in CPU lattice processing
void LatticeProcessor::GetHostData(Token** toks_buf, int** toks_fr_sidx,
                                   LatLink** arcs_buf, int** arcs_fr_size) {
  *toks_fr_sidx = toks_bpr_fr_sidx_h;
  *toks_buf = toks_bpr_h;
  *arcs_fr_size = arcs_apr_fr_size_h; // max_len len
  *arcs_buf = arcs_apr_h; // start of max_len len arcs
}

// CudaLatticeDecoder Implementation
// constructor
CudaLatticeDecoder::CudaLatticeDecoder(const CudaFst &fst, const TransitionModel &trans_model,
                                       const CudaLatticeDecoderConfig &config):
  config_(config), fst_(fst), trans_model_(trans_model), bytes_cuda_malloc(0) {
  KALDI_VLOG(1) << "CudaLatticeDecoder Constructor\n";
  int32 device;
  cudaGetDevice(&device);
  CU_SAFE_CALL(cudaGetLastError());

  // for CUDA_PRINTF
  if (config_.verbose > 4) cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1e7);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  // GPU utilization
  total_threads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount *
                  config.gpu_fraction;

  bytes_cuda_malloc += histogram_prev_toks_.Allocate(config_.beam,
                       (int32)(config_.beam * 0.5), 1.0);

  cudaEventCreateWithFlags(&event_pt, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&event_ll, cudaEventDisableTiming);

  cudaStreamCreateWithFlags(&stream_comp, cudaStreamNonBlocking);
  for (int32 i = 0; i < LAT_BUF_SIZE; i++)
    cudaStreamCreateWithFlags(&stream_lat[i], cudaStreamNonBlocking);
  cudaStreamCreateWithPriority(&stream_ll, cudaStreamNonBlocking, -1);

  cudaMalloc(&ne_queue_d, sizeof(int32)*config.max_tokens_per_frame);
  bytes_cuda_malloc += sizeof(int32);
  cudaMalloc(&barrier_d, sizeof(int32)); bytes_cuda_malloc += sizeof(int32);

  cudaMemset(barrier_d, 0, sizeof(int32));
  CU_SAFE_CALL(cudaGetLastError());

  cudaMalloc(&cutoff_d, sizeof(CostType)); bytes_cuda_malloc += sizeof(CostType);
  cudaMalloc(&cutoff_prev_d, sizeof(CostType));
  bytes_cuda_malloc += sizeof(CostType);
  cudaMalloc((void**)&num_arcs_till_last_d, sizeof(int32));
  bytes_cuda_malloc += sizeof(int32);
  cudaMalloc(&modified_d, sizeof(int32) * 2);
  bytes_cuda_malloc += sizeof(int32) * 2;

  cudaMalloc((void**)&current_tokens_lookup_d,
             sizeof(TokenLookupElem)*fst_.NumStates());
  bytes_cuda_malloc += sizeof(TokenLookupElem) * fst_.NumStates();

  // for pruning
  bytes_cuda_malloc += lattice_processor_.Allocate(config.max_tokens_per_frame,
                       config.max_lat_arc_per_frame, config.max_len,
                       config.max_tokens, config.max_arcs, fst_);

  lat_arcs_buf_.Allocate(config.max_arcs, NULL, NULL, NULL,
                         lattice_processor_.GetDeviceArcsBpr());
  cudaMalloc((void**)&lat_arcs_buf_end_, sizeof(int));
  bytes_cuda_malloc += lat_arcs_buf_.GetCudaMallocBytes() + sizeof(int);

  for (int32 j = 0; j < LAT_BUF_SIZE; j++) {
    lat_toks_bufs_[j].Allocate(config.max_tokens_per_frame);
    bytes_cuda_malloc += lat_toks_bufs_[j].GetCudaMallocBytes();
  }

  // In each frame, we save the token
  // information in an array whose size is the number of arcs. This
  // ensures there are no write conflicts between threads since each
  // arc can be accessed at most once in each frame. It's a temp solution
  cudaMalloc((void**)&token_per_arc_d, sizeof(Token)*config.max_lat_arc_per_frame);
  cudaMalloc((void**)&token_per_arc_update_d,
             sizeof(int32)*config.max_lat_arc_per_frame); // temp solution
  cudaMemset(token_per_arc_update_d, 0,
             sizeof(int32)*config.max_lat_arc_per_frame); // temp solution
  bytes_cuda_malloc += (sizeof(Token) + sizeof(int32)) *
                       (config.max_lat_arc_per_frame);

  // scan & expand
  cudaMalloc((void**)&d_q_token_from_narcs, sizeof(int));
  cudaMalloc((void**)&d_block_sums_scan, sizeof(int)*config.max_tokens_per_frame);
  cudaMalloc((void**)&d_n_CTA_done, sizeof(int));
  cudaMalloc((void**)&d_block_sums, sizeof(int)*config.max_tokens_per_frame);
  cudaMalloc((void**)&d_degrees_scan, sizeof(int)*config.max_tokens_per_frame);
  cudaMalloc((void**)&d_q_arc_offset, sizeof(int)*config.max_tokens_per_frame);
  bytes_cuda_malloc += sizeof(int32) * (2 + 4 * config.max_tokens_per_frame);

  num_frames_decoded_ = 0;
  UpdateTokPointersByFrame(num_frames_decoded_);

  cudaStreamSynchronize(stream_comp);
  cudaStreamSynchronize(stream_lat[0]);
  cudaStreamSynchronize(cudaStreamPerThread);
  // sgemm requires shared memory and we don't want cache config changing.
  // So set a device wide cache config.
  cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);

  const std::vector<int32>& id2pdf_id = trans_model_.GetId2pdf();
  int data_size = id2pdf_id.size() * sizeof(int);
  cudaMalloc((void**)&id2pdf_d_, data_size);
  bytes_cuda_malloc+= data_size;
  cudaMemcpy(id2pdf_d_, id2pdf_id.data(), data_size, cudaMemcpyHostToDevice);

  if (config_.verbose > 1)
    GetFreeMemoryStat("After initlization:");
}

CudaLatticeDecoder::~CudaLatticeDecoder() {
  KALDI_VLOG(1) << "CUDA LatticeDecoder DESTRUCTOR\n";

  if (config_.verbose > 1)
    GetFreeMemoryStat("End of decoding:");

  for (int32 j = 0; j < LAT_BUF_SIZE; j++) lat_toks_bufs_[j].Free();
  lat_arcs_buf_.Free(true);
  cudaFree(lat_arcs_buf_end_);
  lattice_processor_.Free();
  histogram_prev_toks_.Free();

  cudaFree(current_tokens_lookup_d);

  cudaFree(ne_queue_d);
  cudaFree(barrier_d);

  cudaFree(cutoff_d);
  cudaFree(cutoff_prev_d);
  cudaFree(num_arcs_till_last_d);
  cudaFree(modified_d);

  cudaFree(token_per_arc_d);
  cudaFree(token_per_arc_update_d);

  cudaFree(id2pdf_d_);

  cudaEventDestroy(event_pt);
  cudaEventDestroy(event_ll);
  cudaStreamDestroy(stream_comp);
  for (int32 i = 0; i < LAT_BUF_SIZE; i++)
    cudaStreamDestroy(stream_lat[i]);
  cudaStreamDestroy(stream_ll);
}

// initialize parameters routine for launching cuda kernel
// GPU holds a local version of processTokens_params struct during launching
void CudaLatticeDecoder::InitParams(processTokens_params* params) {
  params->prev_toks = (*prev_toks_);
  params->cur_toks = (*cur_toks_);
  params->current_tokens_lookup = current_tokens_lookup_d;
  params->cutoff = cutoff_d;
  params->cutoff_prev = cutoff_prev_d;
  params->lat_arcs_sub_vec = lat_arcs_buf_;
  params->lat_arcs_buf_end = lat_arcs_buf_end_;
  params->token_per_arc = token_per_arc_d;
  params->token_per_arc_update = token_per_arc_update_d;

  params->lattice_processor = lattice_processor_;
  params->histogram_prev_toks = histogram_prev_toks_;

  params->e_offsets = fst_.e_offsets_d;
  params->ne_offsets = fst_.ne_offsets_d;
  params->arc_ilabels = fst_.arc_ilabels_d;
  params->arc_olabels = fst_.arc_olabels_d;
  params->arc_weights = fst_.arc_weights_d;
  params->arc_nextstates = fst_.arc_nextstates_d;

  params->cuda_decodable = cuda_decodable_;
  params->modified = modified_d;
  params->ne_queue = ne_queue_d;
  params->barrier = barrier_d;

  params->beam = config_.beam;
  params->verbose = config_.verbose;
  params->lattice_beam = config_.lattice_beam;
  params->max_len = config_.max_len;
  params->numArcs = fst_.NumArcs();
  params->frame = num_frames_decoded_;
  params->num_arcs_till_last = num_arcs_till_last_d;
  params->max_lat_arc_per_frame = config_.max_lat_arc_per_frame;
  params->max_active = config_.max_active;

  params->d_q_token_from_narcs = d_q_token_from_narcs;
  params->d_block_sums_scan = d_block_sums_scan;
  params->d_n_CTA_done = d_n_CTA_done;
  params->d_block_sums = d_block_sums;
  params->d_degrees_scan = d_degrees_scan;
  params->d_q_arc_offset = d_q_arc_offset;
}

// call InitDecoding if you have already decoded an
// utterance and want to start with a new utterance.
void CudaLatticeDecoder::InitDecoding() {
  if (config_.verbose > 1 ) KALDI_LOG << "CUDA LatticeDecoder InitDecoding\n";
  num_frames_decoded_ = 0;
  for (int32 i = 0; i < LAT_BUF_SIZE; i++) {
    ClearToks(lat_toks_bufs_[i]);
  }
  lat_arcs_buf_.Clear();

  UpdateTokPointersByFrame(num_frames_decoded_);
  lattice_processor_.Initialize();
  CU_SAFE_CALL(cudaGetLastError());

  // we launch 256 threads as a block, i.e. 8 cooperative_groups
  // in cuda kernel of dynamic load balancing. more details are described there
  // we use a static launch size to reduce the kernel launch time 30us->10us
  int32 threads = PROC_DIMX;
  int32 blocks = DIV_ROUND_UP(total_threads, threads);

  // try to reduce number of tokens_allocation by not doing allocate, but only
  // init TokenLookupElem

  _initialize_all_states <<< blocks, threads, 0, stream_comp>>>(
    current_tokens_lookup_d, fst_.NumStates(), barrier_d);
  CU_SAFE_CALL(cudaGetLastError());


  // initialize decoding:
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);

  processTokens_params params;
  InitParams(&params);
  _add_initial_token <<< 1, 1, 0, stream_comp>>>(params, start_state);
  CU_SAFE_CALL(cudaGetLastError());

  ProcessNonemitting();

  if (config_.verbose > 1 ) KALDI_LOG <<
                                        "end of CUDA LatticeDecoder InitDecoding\n";
}

void CudaLatticeDecoder::UpdateTokPointersByFrame(uint32 frame) {
  cur_toks_ = &lat_toks_bufs_[frame % LAT_BUF_SIZE];
  prev_toks_ = &lat_toks_bufs_[(frame - 1) % LAT_BUF_SIZE];
  // single buffer in lat_arcs_buf_, so it doesn't need to do this
}

void CudaLatticeDecoder::ClearToks(TokenMergeVector &toks) {
  // cannot actually delete tokens as they are still used as lattice node
  toks.Clear(stream_comp);
}

void CudaLatticeDecoder::PreProcessTokens() {
  num_frames_decoded_++;
  UpdateTokPointersByFrame(num_frames_decoded_);
  // ClearToks(*cur_toks_); // to reduce a kernel
  // dont need to clear arcs as we directly take the final buffer into this vector
}

void CudaLatticeDecoder::ProcessTokens() {
  PUSH_RANGE("ProcessTokens", 2);
  KALDI_VLOG(4) << num_frames_decoded_ << std::endl;

  // we launch 256 threads as a block, i.e. 8 cooperative_groups
  // in cuda kernel of dynamic load balancing. more details are described there
  // we use a static launch size to reduce the kernel launch time 30us->10us
  dim3 threads(PROC_DIMX, 1);
  dim3 blocks(max(1, DIV_ROUND_UP(total_threads, (threads.x * threads.y))));
  if (num_frames_decoded_ == 1) KALDI_VLOG(2) << "# of blocks: " << blocks.x <<
        std::endl;

  // make sure log likelihoods are on the device before starting these kernels
  cudaStreamWaitEvent(stream_comp, event_ll, 0);
  processTokens_params params;
  InitParams(&params);
  _process_tokens<<<blocks, threads, 0, stream_comp>>>(params);
  CU_SAFE_CALL(cudaGetLastError());

  cudaEventSynchronize(event_pt); // wait for last frame to finish
  cudaEventRecord(event_pt, stream_comp);

  POP_RANGE;
}

void CudaLatticeDecoder::ProcessNonemitting() {
  PUSH_RANGE("ProcessNonemitting", 0);

  // we launch 256 threads as a block, i.e. 8 cooperative_groups
  // in cuda kernel of dynamic load balancing. more details are described there
  // we use a static launch size to reduce the kernel launch time 30us->10us
  dim3 threads(PROC_DIMX, 1);
  dim3 blocks(max(1, DIV_ROUND_UP(total_threads, (threads.x * threads.y))));

  processTokens_params params;
  InitParams(&params);
  _process_tokens<<<blocks, threads, 0, stream_comp>>>(params, true);
  CU_SAFE_CALL(cudaGetLastError());

  POP_RANGE;
}


void CudaLatticeDecoder::DecodeChunk(CuMatrix<BaseFloat> *post_chunk) {
  int chunk_used_len=0;
  while (chunk_used_len < post_chunk->NumRows()) {
    // one frame from the chunk
    cuda_decodable_ = CuMatrixScaledMapper(id2pdf_d_,
                      config_.acoustic_scale, post_chunk->Row(chunk_used_len).Data()); 

    PreProcessTokens(); 
    ProcessTokens(); 

    chunk_used_len++;
    // if utt is too long, discard the tail
    if (num_frames_decoded_ + 1 >= config_.max_len) break;
  }
}

void CudaLatticeDecoder::Decode(MatrixChunker *decodable) {
  while ( !decodable->IsLastFrame(num_frames_decoded_ - 1)) {
    bool last_frame = decodable->IsLastFrame(num_frames_decoded_ - 0);
    if (num_frames_decoded_ + 1 >= config_.max_len) {
      last_frame = true;
      KALDI_WARN << "the utterance is too long, cutoff after frames: " 
                 << num_frames_decoded_ + 1;
    }

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

// GPU lattice prune and copy the processed lattice nodes and arcs to host
void CudaLatticeDecoder::FinalProcessLattice(Token** toks_buf, int** toks_fr_sidx,
    LatLink** arcs_buf, int** arcs_fr_size, TokenMergeVector** toks_vec_last_fr, 
    int *num_frames_decoded) {
  PUSH_RANGE("FinalProcessLattice", 3);

  cudaStreamSynchronize(stream_comp); // after fini comp. we can start copy
  // copy unpruned toks to host
  lattice_processor_.CopyToksToHost(num_frames_decoded_, stream_lat[0]);
  // GPU lattice pruning
  PruneActiveTokens(stream_comp, stream_comp, config_.lat_fraction);
  // copy the TokenState vector in the last frame, used by ComputeFinalCosts()
  CU_SAFE_CALL(cudaGetLastError());
  (*cur_toks_).CopyDataToHost(stream_lat[1]);
  *toks_vec_last_fr = cur_toks_;
  cudaStreamSynchronize(stream_comp); // wait for lattice pruning
  // copy pruned lattice arcs to host
  lattice_processor_.CopyArcsToHost(num_frames_decoded_, stream_lat[1]);
  // wait for all streams finishing
  cudaStreamSynchronize(stream_lat[0]);
  cudaStreamSynchronize(stream_lat[1]);
  // get host data from lattice_processor_, used by CPU lattice processing
  lattice_processor_.GetHostData(toks_buf, toks_fr_sidx,
                                 arcs_buf, arcs_fr_size);
  CU_SAFE_CALL(cudaGetLastError());

  *num_frames_decoded = num_frames_decoded_;

  KALDI_VLOG(1) << "Average tokens number, total frame: "
                << (*toks_fr_sidx)[num_frames_decoded_ + 1] / num_frames_decoded_
                << ", " << num_frames_decoded_;
  POP_RANGE;
}

void CudaLatticeDecoder::PruneActiveTokens(cudaStream_t wait_st,
    cudaStream_t run_st, BaseFloat gpu_ratio) {
  // we launch 256 threads as a block, i.e. 8 cooperative_groups
  // in cuda kernel of dynamic load balancing. more details are described there
  // we use a static launch size to reduce the kernel launch time 30us->10us
  dim3 threads(PROC_DIMX, 1);
  dim3 blocks(max(1, (int)DIV_ROUND_UP(total_threads * gpu_ratio,
                                       (threads.x * threads.y))));
  cudaStreamSynchronize(wait_st);
  if (config_.verbose > 1) KALDI_LOG << "PruneActiveTokens, # of blocks: " <<
                                       blocks.x << std::endl;
  processTokens_params params;
  InitParams(&params);
  _prune_active_tokens <<< blocks, threads, 0, run_st>>>(params);
}

// Outputs an FST corresponding to the single best path
// through the lattice. In lattice decoder, it is deprecated
bool CudaLatticeDecoder::GetBestPath(Lattice *fst_out,
                                     bool use_final_probs) const {
  KALDI_ERR << "We don't have this implementation in lattice decoder";
  return false;
}


} // end namespace kaldi.
