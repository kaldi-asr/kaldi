// decoder/cuda-decoder-utils.h

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

#ifndef KALDI_CUDA_DECODER_UTILS_H_
#define KALDI_CUDA_DECODER_UTILS_H_


#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "omp.h"
#include "cuda_runtime.h"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <float.h>
#include <math.h>
#include <cooperative_groups.h>
#include "math_constants.h"
#include "omp.h"

#include "util/stl-utils.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"

namespace kaldi {

// we put functions and macros shared by different modules in this file

// cuda macro definition

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__

#else
#define HOST
#define DEVICE
#endif

#define USE_NVTX
#ifdef USE_NVTX
#include "nvToolsExt.h"
const uint32 colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 
                         0x0000ffff, 0x00ff0000, 0x00ffffff};
const int32 num_colors = sizeof(colors) / sizeof(uint32);

#define PUSH_RANGE(name,cid) do { \
    int32 color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
} while (0);
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

// decoder macro

#define __DEBUG__
#ifdef __DEBUG__
#define VERBOSE 5
#define CUDA_PRINTF(format,...) printf(format, ##__VA_ARGS__)
#else
#define VERBOSE 0
#define CUDA_PRINTF(format,...)
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
#define MEMADVISE // used after Pascal, details in: 
// http:// mug.mvapich.cse.ohio-state.edu/static/media/mug/presentations/2016/MUG16_GPU_tutorial_V5.pdf
#endif

#define DIV_ROUND_UP(a,b) ((a+b-1)/b)

// decode & encode function of tok address, used in host & device
// pack frame & index into uint64 as the address of a tok
#define ENCODE_TOK_IDX_PAIR(frame,idx) (((uint64)(frame)<<32)+(idx))
// get frame & index in the per-frame vector of a tok address packed in uint64
#define DECODE_TOK_IDX_PAIR(frame,idx,v) { \
    frame=(((uint64)v)>>32); \
    idx=(((uint64)v)&(((uint64)1<<32)-1)); \
}

// host function definition

// get current GPU memory usage
void get_free_memory_stat(const char *prefix);

// a combination of cudaMallocManaged & cudaMemAdvise
void cuda_malloc_managed_preferred_device(void** devPtr, size_t size);

// inline host device function definition

// In atomic based token recombination, we pack the
// cost and the arc index into an uint64 to represent the token
// before recombination, with the former one in the higher bits
// for comparison purpose.
// for speedup purpose, make them inline (5% 0.165->0.158)
inline HOST DEVICE uint64 pack_cost_idx_into_uint64(BaseFloat cost, int32 idx) {
  uint32 i_cost = *(uint32 *) & cost;
  if (i_cost & 0x80000000)
    i_cost = i_cost ^ 0xFFFFFFFF;
  else
    i_cost = i_cost ^ 0x80000000;
  return (uint64)i_cost << 32 | idx;
}

// Unpacks a cost
inline HOST DEVICE BaseFloat unpack_cost_from_uint64(uint64 packed) {
  uint32 i_cost = packed >> 32;
  if (i_cost & 0x80000000)
    i_cost = i_cost ^ 0x80000000;
  else
    i_cost = i_cost ^ 0xFFFFFFFF;
  return *(BaseFloat *) & i_cost;
}

// Unpacks a idx for tracing the data
inline HOST DEVICE int32 unpack_idx_from_uint64(uint64 packed) {
  // assert (!(packed & 0x80000000));
  return packed & 0x7FFFFFFF;
}

// inline device function definition
#ifdef __CUDACC__

// fast load 16 bits using CUDA ASM
inline  DEVICE void fast_load16(void *a, const void *b) {
  const ulong2 *src = reinterpret_cast<const ulong2*>(b);
  ulong2 &dst = *reinterpret_cast<ulong2*>(a);
  asm("ld.global.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(src));
}

// fast store 16 bits using CUDA ASM
inline  DEVICE void fast_store16(void *a, const void *b) {
  const ulong2 src = *reinterpret_cast<const ulong2*>(b);
  asm("st.global.v2.u64 [%0], {%1,%2};" :: "l"(a), "l"(src.x), "l"(src.y));
}

// fast store 8 bits using CUDA ASM
inline  DEVICE void fast_store8(void *a, const void *b) {
#if 0
  memcpy(a, b, 8);
#else
  *(uint64*)a = (*(uint64*)b);
#endif
}

// TODO: we need a fast 32 bits storing function
inline  DEVICE void fast_store32(void *a, const void *b) {
  memcpy(a, b, 32);
}

// overload CUDA atomicMin to consume double
inline DEVICE void atomic_min(double *address, double val) {
  unsigned long long *address_ull = (unsigned long long *)address;
  double minval = *address;
  while (val < minval) {  // if my value is less than minimum
    minval = val;         // update the minimum to my value locally
    // write minimum and read back value
    val = __longlong_as_double(atomicExch(address_ull, __double_as_longlong(val)));
  } // if the new value is < the minimum I wrote I need to try again.
}

// overload CUDA atomicMin to consume BaseFloat
inline DEVICE void atomic_min(BaseFloat *address, BaseFloat val) {
  uint32 *address_ui = (uint32  *)address;
  BaseFloat minval = *address;
  while (val < minval) {  // if my value is less than minimum
    minval = val;         // update the minimum to my value locally
    // write minimum and read back value
    val = __uint_as_float(atomicExch(address_ui, __float_as_uint(val)));
  } // if the new value is < the minimum I wrote I need to try again.
}

// Assumptions: 1-d grid and blocks. No threads "early-exit" the grid.
// No stream priorities
static DEVICE inline void _grid_sync(volatile int *fast_epoch) {
  __syncthreads();
  if (threadIdx.x == 0) {
    // gridDim.x-1 blocks are adding 1
    // and one block is adding 0x80000000 - (gridDim.x-1)
    // so the whole sum is 0x80000000
    int nb = 1;
    if (blockIdx.x == 0) {
      nb = 0x80000000 - (gridDim.x - 1);
    }
    int old_epoch = *fast_epoch;
    __threadfence();
    atomicAdd((int*)fast_epoch, nb);
    // wait for the sign bit to commute
    int cnt = 0;
    while (((*fast_epoch) ^ old_epoch) >= 0) ;
  }
  __syncthreads();
}

DEVICE inline void grid_sync(int *barrier) {
  _grid_sync((volatile int*)barrier);
}

#endif

// class definition

// We don't use cub::DeviceHistogram because its kernel launch overhead is 30us
// while current GPU decoding takes around 200 us to decode 1 frame
#define MAX_HISTOGRAM_SIZE 10
class CudaHistogram {
  public:
    int32 Allocate(BaseFloat beam, BaseFloat beam_lowest, BaseFloat step);
    void Free();

#ifdef __CUDACC__
    inline DEVICE void Initialize(BaseFloat best_cost) {
        *best_cost_ = best_cost;
    }
    inline DEVICE int32 Size() const { return (beam_ - beam_lowest_); }
    inline DEVICE void AddScore2LocalHist(BaseFloat cost, int32 *hist_local) {
        int32 dist = (int)(cost - *best_cost_);
        assert(dist <= beam_);
        if (dist <= beam_lowest_) hist_local[0]++;
        else if (dist == beam_) hist_local[Size() - 1]++;
        else hist_local[(int32)(dist - beam_lowest_)]++;
    }
    inline DEVICE void AggregateLocalHist(int32 *hist_local) {
        for (int i = 0; i < Size(); i++) {
            if (hist_local[i] != 0)
                atomicAdd(hist_global_ + i, hist_local[i]);
        }
    }
    inline DEVICE void GetCutoff(BaseFloat *cutoff_from_hist, int32 cutoff_num,
                                 int verbose = 0) {
        int32 acc = 0, i = 0;
        for (i = 0; i < Size(); i++) {
            acc += hist_global_[i];
            if (acc > cutoff_num) break;
        }
        BaseFloat ret_beam = i + beam_lowest_;
        *cutoff_from_hist = *best_cost_ + ret_beam;
        if (verbose > 2) {
            CUDA_PRINTF("hist_LF %f %i\n", *cutoff_from_hist, acc);
        }
        memset(hist_global_, 0, Size());
    }
#endif

  private:
    // configuration
    BaseFloat beam_;
    BaseFloat beam_lowest_;
    BaseFloat step_;
    // global cache data
    BaseFloat* best_cost_;
    int32* hist_global_;
};

// WFST struct designed for GPU memory
class CudaFst {
  public:
    typedef fst::StdArc StdArc;
    typedef StdArc::StateId StateId;
    typedef BaseFloat CostType;
    typedef StdArc::Weight StdWeight;
    typedef StdArc::Label Label;

    CudaFst() {};
    void Initialize(const fst::Fst<StdArc> &fst);
    void Finalize();

    uint32 NumStates() const {  return numStates; }
    uint32 NumArcs() const {  return numArcs; }
    StateId Start() const { return start; }
    HOST DEVICE BaseFloat Final(StateId state) const;
    size_t GetCudaMallocBytes() const { return bytes_cudaMalloc; }

    uint32 numStates;               // total number of states
    uint32 numArcs;               // total number of states
    StateId  start;

    uint32 max_ilabel;              // the largest ilabel
    uint32 e_count, ne_count,
           arc_count;       // number of emitting and non-emitting states

    // This data structure have 2 matrices (one emitting one non-emitting).
    // Offset arrays are numStates+1 in size.
    // Arc values for state i are stored in the range of [i,i+1)
    // size numStates+1
    uint32 *e_offsets_h, *e_offsets_d;              // Emitting offset arrays
    uint32 *ne_offsets_h, *ne_offsets_d;            // Non-emitting offset arrays

    // These are the values for each arc. Arcs belonging to state i are found 
    // in the range of [offsets[i], offsets[i+1]) (Size arc_count+1)
    BaseFloat *arc_weights_h, *arc_weights_d;
    StateId *arc_nextstates_h, *arc_nextstates_d;
    int32 *arc_ilabels_h, *arc_ilabels_d, *arc_olabels_d;
    int32 *arc_olabels_h;

    // final costs
    BaseFloat *final_h, *final_d;
    // allocation size
    size_t bytes_cudaMalloc;
};

} // end namespace kaldi.

#endif
