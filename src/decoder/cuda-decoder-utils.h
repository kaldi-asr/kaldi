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


#include <algorithm>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <float.h>
#include <math.h>
#include <cooperative_groups.h>

#include "util/stl-utils.h"
#include "hmm/transition-model.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-matrix.h"
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
                         0x0000ffff, 0x00ff0000, 0x00ffffff
                        };
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
#define VERBOSE 2
#define CUDA_PRINTF(VB, format,...) if (VERBOSE > VB) printf( format, ##__VA_ARGS__)
#else
#define VERBOSE 0
#define CUDA_PRINTF(VB, format,...)
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
void GetFreeMemoryStat(const char *prefix);

// a combination of cudaMallocManaged & cudaMemAdvise
void CudaMallocManagedPreferredDevice(void** devPtr, size_t size);

// inline host device function definition

union float_uint {
  float f;
  uint32 u;
};
inline HOST DEVICE uint32 float_as_uint(BaseFloat val) {
  return ((float_uint*)&val)->u;
}
inline HOST DEVICE BaseFloat uint_as_float(uint32 val) {
  return ((float_uint*)&val)->f;
}

// In atomic based token recombination, we pack the
// cost and the arc index into an uint64 to represent the token
// before recombination, with the former one in the higher bits
// for comparison purpose.
// for speedup purpose, make them inline (5% 0.165->0.158)
inline HOST DEVICE uint64 pack_cost_idx_into_uint64(BaseFloat cost, int32 idx) {
  uint32 i_cost = float_as_uint(cost);
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
  return uint_as_float(i_cost);
}

// Unpacks a idx for tracing the data
inline HOST DEVICE int32 unpack_idx_from_uint64(uint64 packed) {
  return packed & 0x7FFFFFFF;
}

// inline device function definition
#ifdef __CUDACC__

// another choice is pre-store the result of this binsearch, e.g. do something like:
// for (int j=0; j < params->d_degrees[idx]; j++)
//  params->d_lowerbound[j + params->d_degrees_scan[idx]] = idx;
// however this for loop costs 9% more time, while the binsearch costs 6% time
// thus the fastest way is doing binsearch without pre-store
inline DEVICE int binsearch_maxle(const int *vec, const int val, int low,
                                  int high) {
  while (true) {
    if (low == high)
      return low; //we know it exists
    if ((low + 1) == high)
      return (vec[high] <= val) ? high : low;

    int mid = low + (high - low) / 2;

    if (vec[mid] > val)
      high = mid - 1;
    else
      low = mid;
  }
}


// fast load 16 bytes using CUDA ASM
inline  DEVICE void fast_load16(void *a, const void *b) {
  const ulong2 *src = reinterpret_cast<const ulong2*>(b);
  ulong2 &dst = *reinterpret_cast<ulong2*>(a);
  asm("ld.global.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(src));
}

// fast store 16 bytes using CUDA ASM
inline  DEVICE void fast_store16(void *a, const void *b) {
  const ulong2 src = *reinterpret_cast<const ulong2*>(b);
  asm("st.global.v2.u64 [%0], {%1,%2};" :: "l"(a), "l"(src.x), "l"(src.y));
}

// fast store 8 bytes using CUDA ASM
inline  DEVICE void fast_store8(void *a, const void *b) {
  *(uint64*)a = (*(uint64*)b);
}

// TODO: we need a fast 32 bytes storing function
inline  DEVICE void fast_store32(void *a, const void *b) {
  memcpy(a, b, 32);
}

// overload CUDA atomicMin to consume double
inline DEVICE double atomic_min(double *address, double val) {
  unsigned long long *address_ull = (unsigned long long *)address;
  double minval = *address;
  while (val < minval) {  // if my value is less than minimum
    minval = val;         // update the minimum to my value locally
    // write minimum and read back value
    val = __longlong_as_double(atomicExch(address_ull, __double_as_longlong(val)));
  } // if the new value is < the minimum I wrote I need to try again.
  return minval;
}

// overload CUDA atomicMin to consume BaseFloat
inline DEVICE BaseFloat atomic_min(BaseFloat *address, BaseFloat val) {
  uint32 *address_ui = (uint32  *)address;
  BaseFloat minval = *address;
  while (val < minval) {  // if my value is less than minimum
    minval = val;         // update the minimum to my value locally
    // write minimum and read back value
    val = __uint_as_float(atomicExch(address_ui, __float_as_uint(val)));
  } // if the new value is < the minimum I wrote I need to try again.
  return minval;
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
    int32 dist = (int)(cost - *best_cost_) / step_;
    assert(dist <= beam_);
    if (dist <= beam_lowest_) hist_local[0]++;
    else if (dist == beam_) hist_local[Size() - 1]++;
    else hist_local[(int32)(dist - beam_lowest_)]++;
  }
  inline DEVICE void AggregateLocalHist(int32 *hist_local) {
    for (int i = 0; i < Size(); i++) {
      if (hist_local[i] != 0)
        // "fire and forget" atomics
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
    BaseFloat ret_beam = step_ * (i + beam_lowest_);
    *cutoff_from_hist = *best_cost_ + ret_beam;
    if (verbose > 2) {
      CUDA_PRINTF(2, "hist_LF %f %i\n", *cutoff_from_hist, acc);
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

class MatrixChunker {
#define DEC_CHUNK_BUF_SIZE 2
 public:
  // This constructor creates an object that will not delete "likes"
  // when done.
  MatrixChunker(const Matrix<BaseFloat> &likes, int chunk_len): likes_(&likes),
    delete_likes_(false), chunk_len_(chunk_len), chunk_id_(0) {
  }

  int32 NumFramesReady() const { return likes_->NumRows(); }

  bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

  void LogLikelihoodChunk(int32 frame, CuMatrix<BaseFloat>** out, cudaStream_t stream) {
    if (frame >= likes_->NumRows()) return;
    int len = std::min(chunk_len_, likes_->NumRows() - frame);
    assert(len);
    CuMatrix<BaseFloat>& loglike_d = loglikes_d[++chunk_id_ % DEC_CHUNK_BUF_SIZE];
    int data_size = likes_->NumCols() * sizeof(BaseFloat);
    CU_SAFE_CALL(cudaGetLastError());
    // we seldom Resize()
    if (loglike_d.NumRows() != len) loglike_d.Resize(len, likes_->NumCols(), kUndefined);
    // we need cudaMemcpyAsync with stream and cannot use kaldi::CopyFromMat
    // as they have different strides, we have to do this
    cudaMemcpy2DAsync(loglike_d.Data(), loglike_d.Stride()*sizeof(BaseFloat), likes_->Row(frame).Data(), 
                      likes_->Stride()*sizeof(BaseFloat), data_size, len, cudaMemcpyHostToDevice, stream);
    CU_SAFE_CALL(cudaGetLastError());
    *out = &loglike_d;
    return;
  };

  ~MatrixChunker() {
    if (delete_likes_) delete likes_;
  }
  
  const Matrix<BaseFloat> *likes_;
  bool delete_likes_;
  CuMatrix<BaseFloat> loglikes_d[DEC_CHUNK_BUF_SIZE];

  int chunk_len_, chunk_id_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(MatrixChunker);
};

class CuMatrixScaledMapper {
 public:
  CuMatrixScaledMapper() : id2pdf_d_(NULL), acoustic_scale_(0),
    loglike_d_(NULL) {}
  CuMatrixScaledMapper(int32 *id2pdf_d, BaseFloat acoustic_scale,
                                BaseFloat* loglike_d) : id2pdf_d_(id2pdf_d),
    acoustic_scale_(acoustic_scale), loglike_d_(loglike_d) {}
  DEVICE BaseFloat LogLikelihood(int32 tid) const {
    assert(id2pdf_d_);
    int idx = id2pdf_d_[tid];
    return loglike_d_[idx] * acoustic_scale_;
  }
 private:
  int32 *id2pdf_d_;
  BaseFloat acoustic_scale_, *loglike_d_;
};

} // end namespace kaldi.

#endif
