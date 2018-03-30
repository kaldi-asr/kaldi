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

// cuda macro

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

// #define __DEBUG__
#ifdef __DEBUG__
#define VERBOSE 5
#define CUDA_PRINTF(format,...) printf(format, ##__VA_ARGS__)
#else
#define VERBOSE 0
#define CUDA_PRINTF(format,...)
#endif

// #define MEMADVISE // used after Pascal, details: 
// http:// mug.mvapich.cse.ohio-state.edu/static/media/mug/presentations/2016/MUG16_GPU_tutorial_V5.pdf

#define DIV_ROUND_UP(a,b) ((a+b-1)/b)

// decode & encode function of tok address, used in host & device
// pack frame & index into uint64 as the address of a tok
#define ENCODE_TOK_IDX_PAIR(frame,idx) (((uint64)(frame)<<32)+(idx))
// get frame & index in the per-frame vector of a tok address packed in uint64
#define DECODE_TOK_IDX_PAIR(frame,idx,v) { \
    frame=(((uint64)v)>>32); \
    idx=(((uint64)v)&(((uint64)1<<32)-1)); \
}

namespace kaldi {


// Assumptions: 1-d grid and blocks. No threads "early-exit" the grid.
// No stream priorities
DEVICE void __grid_sync_nv_internal(int32 *barrier);

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
