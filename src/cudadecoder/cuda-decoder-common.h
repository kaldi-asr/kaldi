// cudadecoder/cuda-decoder-common.h
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

#ifndef KALDI_CUDA_DECODER_CUDA_DECODER_UTILS_H_
#define KALDI_CUDA_DECODER_CUDA_DECODER_UTILS_H_
#include "cudamatrix/cu-common.h"
#include "util/stl-utils.h"

// A decoder channel is linked to one utterance. Frames
// from the same must be sent to the same channel.
//
// A decoder lane is where the computation actually happens
// a decoder lane is given a frame and its associated channel
// and does the actual computation
//
// An analogy would be lane -> a core, channel -> a software thread

// Some config parameters can be computed using other parameters
// (e.g. we can set main_q_capacity using max-active)
// Those values are the different factors between parameters that we know
// and parameters we want to set
#define KALDI_CUDA_DECODER_MAX_ACTIVE_MAIN_Q_CAPACITY_FACTOR 4
#define KALDI_CUDA_DECODER_AUX_Q_MAIN_Q_CAPACITIES_FACTOR 3

// If we're at risk of filling the tokens queue,
// the beam is reduced to keep only the best candidates in the
// remaining space
// We then slowly put the beam back to its default value
// beam_next_frame = min(default_beam, RECOVER_RATE * beam_previous_frame)
#define KALDI_CUDA_DECODER_ADAPTIVE_BEAM_RECOVER_RATE 1.2f

// Defines for the cuda decoder kernels
// It shouldn't be necessary to change the DIMX of the kernels

// Below that value, we launch the persistent kernel for NonEmitting
#define KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS 4096

// We know we will have at least X elements in the hashmap
// We allocate space for X*KALDI_CUDA_DECODER_HASHMAP_CAPACITY_FACTOR elements
// to avoid having too much collisions
#define KALDI_CUDA_DECODER_HASHMAP_CAPACITY_FACTOR 1

// Max size of the total kernel arguments
// 4kb for compute capability >= 2.0
#define KALDI_CUDA_DECODER_MAX_KERNEL_ARGUMENTS_BYTE_SIZE (4096)

// When applying the max-active, we need to compute a topk
// to perform that (soft) topk, we compute a histogram
// here we define the number of bins in that histogram
// it has to be less than the number of 1D threads
#define KALDI_CUDA_DECODER_HISTO_NBINS 255

// Number of "heavy duty" process non emitting kernels
// If more non emitting iterations are required, those will be done
// in the one-CTA persistent kernel
#define KALDI_CUDA_DECODER_N_NON_EMITTING_MAIN_ITERATIONS 2

// Adaptive beam parameters
// We will decrease the beam when we detect that we are generating too many
// tokens
// for the first segment of the aux_q, we don't do anything (keep the original
// beam)
// the first segment is made of (aux_q
// capacity)/KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT
// then we will decrease the beam step by step, until 0.
// we will decrease the beam every m elements, with:
// x = (aux_q capacity)/KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT (static
// segment
// y = (aux_q capacity) - x
// m = y / KALDI_CUDA_DECODER_ADAPTIVE_BEAM_NSTEPS
// For more information, please refer to the definition of GetAdaptiveBeam in
// cuda-decoder-kernels.cu
#define KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT 4
#define KALDI_CUDA_DECODER_ADAPTIVE_BEAM_NSTEPS 8
// When applying max_active we don't keep exactly max_active_ tokens,
// but a bit more. And we can call ApplyMaxActiveAndReduceBeam multiple times
// in the first frame (the first times as a pre-filter, the last time at the
// very end of the frame)
// Because keeping a bit more than max_active_ is expected, we add the tolerance
// so that we can avoid triggering ApplyMaxActiveAndReduceBeam for just a few
// tokens above the limit
// at the end of the frame

#define KALDI_CUDA_DECODER_DIV_ROUND_UP(a, b) ((a + b - 1) / b)

#define KALDI_CUDA_DECODER_ASSERT(val, recoverable)                     \
  {                                                                     \
    if ((val) != true) {                                                \
      throw CudaDecoderException("KALDI_CUDA_DECODER_ASSERT", __FILE__, \
                                 __LINE__, recoverable)                 \
    }                                                                   \
  }
// Macro for checking cuda errors following a cuda launch or api call
#ifdef NDEBUG
#define KALDI_DECODER_CUDA_CHECK_ERROR()
#else
#define KALDI_DECODER_CUDA_CHECK_ERROR()                                  \
  {                                                                       \
    cudaError_t e = cudaGetLastError();                                   \
    if (e != cudaSuccess) {                                               \
      throw CudaDecoderException(cudaGetErrorName(e), __FILE__, __LINE__, \
                                 false);                                  \
    }                                                                     \
  }
#endif

#define KALDI_DECODER_CUDA_API_CHECK_ERROR(e)                             \
  {                                                                       \
    if (e != cudaSuccess) {                                               \
      throw CudaDecoderException(cudaGetErrorName(e), __FILE__, __LINE__, \
                                 false);                                  \
    }                                                                     \
  }

#define KALDI_CUDA_DECODER_1D_KERNEL_LOOP(i, n)                \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(offset, th_idx, n) \
  for (int offset = blockIdx.x * blockDim.x, th_idx = threadIdx.x;        \
       offset < (n); offset += blockDim.x * gridDim.x)

#define KALDI_CUDA_DECODER_IS_LAST_1D_THREAD() (threadIdx.x == (blockDim.x - 1))

#define KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.y; i < (n); i += gridDim.y)

#define KALDI_CUDA_DECODER_DIV_ROUND_UP(a, b) ((a + b - 1) / b)

#define KALDI_CUDA_DECODER_1D_BLOCK 256
#define KALDI_CUDA_DECODER_LARGEST_1D_BLOCK 1024
#define KALDI_CUDA_DECODER_ONE_THREAD_BLOCK 1
#define KALDI_CUDA_DECODER_MAX_CTA_COUNT 4096u
#define KALDI_CUDA_DECODER_MAX_CTA_PER_LANE 512u
namespace kaldi {
namespace cuda_decoder {

// Returning the number of CTAs to launch for (N,M) elements to compute
// M is usually the batch size
inline dim3 KaldiCudaDecoderNumBlocks(int N, int M) {
  dim3 grid;
  grid.x = KALDI_CUDA_DECODER_DIV_ROUND_UP(N, KALDI_CUDA_DECODER_1D_BLOCK);
  unsigned int max_CTA_per_lane =
      std::max(KALDI_CUDA_DECODER_MAX_CTA_COUNT / M, 1u);
  grid.x = std::min(grid.x, max_CTA_per_lane);
  grid.y = M;
  return grid;
}

// Use a fixed number of blocks for nlanes
// Using the max number of CTAs possible for each lane,
// according to KALDI_CUDA_DECODER_MAX_CTA_COUNT
// and KALDI_CUDA_DECODER_MAX_CTA_PER_LANE
inline dim3 KaldiCudaDecoderNumBlocks(int nlanes) {
  dim3 grid;
  unsigned int n_CTA_per_lane =
      std::max(KALDI_CUDA_DECODER_MAX_CTA_COUNT / nlanes, 1u);
  if (n_CTA_per_lane == 0) n_CTA_per_lane = 1;
  grid.x = std::min(KALDI_CUDA_DECODER_MAX_CTA_PER_LANE, n_CTA_per_lane);
  grid.y = nlanes;
  return grid;
}

typedef int32 StateId;
typedef float CostType;
// IntegerCostType is the type used in the lookup table d_state_best_cost
// and the d_cutoff
// We use a 1:1 conversion between CostType <--> IntegerCostType
// IntegerCostType is used because it triggers native atomic operations
// (CostType does not)
typedef int32 IntegerCostType;
typedef int32 LaneId;
typedef int32 ChannelId;

// On the device we compute everything by batch
// Data is stored as 2D matrices (BatchSize, 1D_Size)
// For example, for the token queue, (BatchSize, max_tokens_per_frame_)
// DeviceMatrix owns the data but is not used to access it.
// DeviceMatrix is inherited in DeviceLaneMatrix and DeviceChannelMatrix
// those two classes do the same thing, except that they belong either to a
// channel or lane
// that inheritance is done to clarify the code and help debugging
//
// To actually access the data, we should request an view through
// GetView
// That view contains both host cuda code to access the data. It does not own
// the data.
template <typename T>
// if necessary, make a version that always use ncols_ as the next power of 2
class DeviceMatrix {
  T *data_;
  void Allocate() {
    KALDI_ASSERT(nrows_ > 0);
    KALDI_ASSERT(ncols_ > 0);
    KALDI_ASSERT(!data_);
    CU_SAFE_CALL(cudaMalloc((void **)&data_,
                            (size_t)nrows_ * ncols_ * sizeof(*data_)));
    KALDI_ASSERT(data_);
  }
  void Free() {
    KALDI_ASSERT(data_);
    CU_SAFE_CALL(cudaFree(data_));
    data_ = nullptr;
  }

 protected:
  int32 ncols_;
  int32 nrows_;

 public:
  DeviceMatrix() : data_(NULL), ncols_(0), nrows_(0) {}

  virtual ~DeviceMatrix() {
    if (data_) Free();
  }

  void Resize(int32 nrows, int32 ncols) {
    if (data_) Free();
    KALDI_ASSERT(nrows > 0);
    KALDI_ASSERT(ncols > 0);
    nrows_ = nrows;
    ncols_ = ncols;
    Allocate();
  }

  T *MutableData() {
    KALDI_ASSERT(data_);
    return data_;
  }
  // abstract getInterface...
};

template <typename T>
// if necessary, make a version that always use ncols_ as the next power of 2
class HostMatrix {
  T *data_;
  void Allocate() {
    KALDI_ASSERT(nrows_ > 0);
    KALDI_ASSERT(ncols_ > 0);
    KALDI_ASSERT(!data_);
    CU_SAFE_CALL(cudaMallocHost((void **)&data_, (size_t)nrows_ * ncols_ * sizeof(*data_)));
    KALDI_ASSERT(data_);
  }
  void Free() {
    KALDI_ASSERT(data_);
    CU_SAFE_CALL(cudaFreeHost(data_));
    data_ = nullptr;
  }

 protected:
  int32 ncols_;
  int32 nrows_;

 public:
  HostMatrix() : data_(NULL), ncols_(0), nrows_(0) {}

  virtual ~HostMatrix() {
    if (data_) Free();
  }

  void Resize(int32 nrows, int32 ncols) {
    if (data_) Free();
    KALDI_ASSERT(nrows > 0);
    KALDI_ASSERT(ncols > 0);
    nrows_ = nrows;
    ncols_ = ncols;
    Allocate();
  }

  T *MutableData() {
    KALDI_ASSERT(data_);
    return data_;
  }
  // abstract getInterface...
};

// Views of DeviceMatrix
// Those views are created by either DeviceChannelMatrix or
// DeviceLaneMatrix
// We can access the data (the matrix) associated with that
// Device[Channel|Lane]Matrix without owning that data.
// Which means that we can pass those views by copy
// without triggering a cudaFree, for instance.
// Device[Channel|Lane]Matrix owns the data, [Channel|Lane]MatrixInterface just
// gives access to it
// Generating both host and device interfaces
template <typename T>
struct LaneMatrixView {
  T *data_;
  int32 ncols_;
  __host__ __device__ __inline__ T *lane(const int32 ilane) {
    return &data_[ilane * ncols_];
  }
};

template <typename T>
struct ChannelMatrixView {
  T *data_;
  int32 ncols_;
  __host__ __device__ __inline__ T *channel(const int32 ichannel) {
    return &data_[ichannel * ncols_];
  }
};

// Specializing DeviceMatrix into lane and channel variants.
// Helps with code clarity/debugging
template <typename T>
class DeviceLaneMatrix : public DeviceMatrix<T> {
 public:
  LaneMatrixView<T> GetView() { return {this->MutableData(), this->ncols_}; }

  T *lane(const int32 ilane) {
    return &this->MutableData()[ilane * this->ncols_];
  }
};

template <typename T>
class HostLaneMatrix : public HostMatrix<T> {
 public:
  LaneMatrixView<T> GetView() { return {this->MutableData(), this->ncols_}; }

  T *lane(const int32 ilane) {
    return &this->MutableData()[ilane * this->ncols_];
  }
};

template <typename T>
class DeviceChannelMatrix : public DeviceMatrix<T> {
 public:
  ChannelMatrixView<T> GetView() { return {this->MutableData(), this->ncols_}; }
  T *channel(const int32 ichannel) {
    return &this->MutableData()[ichannel * this->ncols_];
  }
};

// InfoToken contains data that needs to be saved for the backtrack
// in GetBestPath/GetRawLattice
// We don't need the token.cost or token.next_state.
struct __align__(8) InfoToken {
  int32 prev_token;
  int32 arc_idx;
  bool IsUniqueTokenForStateAndFrame() {
    // This is a trick used to save space and PCI-E bandwidth (cf
    // preprocess_in_place kernel)
    // This token is associated with a next_state s, created during the
    // processing of frame f.
    // If we have multiple tokens associated with the state s in the frame f,
    // arc_idx < 0 and -arc_idx is the
    // count of such tokens. We will then have to look at another list to read
    // the actually arc_idx and prev_token values
    // If the current token is the only one, prev_token and arc_idx are valid
    // and can be used directly
    return (arc_idx >= 0);
  }

  // Called if this token is linked to others tokens in the same frame (cf
  // comments for IsUniqueTokenForStateAndFrame)
  // return the {offset,size} pair necessary to list those tokens in the
  // extra_prev_tokens list
  // They are stored at offset "offset", and we have "size" of those
  std::pair<int32, int32> GetSameFSTStateTokensList() {
    KALDI_ASSERT(!IsUniqueTokenForStateAndFrame());

    return {prev_token, -arc_idx};
  }
};

// Device function, used to set a in an InfoToken the [offset,size] related to
// InfoToken.GetSameFSTStateTokensList
__device__ __inline__ void SetSameFSTStateTokensList(int32 offset, int32 size,
                                                     InfoToken *info_token) {
  // We always have size > 0
  *info_token = {offset, -size};
}

// Information about the best path head
// Used by partial hypotheses and endpoiting
struct BestPathTracebackHead {
  int index;
  CostType relative_cost;

  void Reset() { index = -1; }
  bool IsSet() { return (index != -1); }
};

// LaneCounters/ChannelCounters
// The counters are all the singular values associated to a lane/channel
// For instance  the main queue size. Or the min_cost of all tokens in that
// queue
// LaneCounters are used during computation
struct LaneCounters {
  // hannel that this lane will compute for the current frame
  ChannelId channel_to_compute;
  // Pointer to the loglikelihoods array for this channel and current frame
  const BaseFloat *loglikelihoods;
  // Contains both main_q_end and narcs
  // End index of the main queue
  // only tokens at index i with i < main_q_end
  // are valid tokens
  // Each valid token the subqueue main_q[main_q_local_offset, main_q_end[ has
  // a number of outgoing arcs (out-degree)
  // main_q_narcs is the sum of those numbers
  // We sometime need to update both end and narcs at the same time using a
  // single atomic,
  // which is why they're packed together
  int2 main_q_narcs_and_end;
  // contains the requested queue length which can
  // be larger then the actual queue length in the case of overflow
  int32 main_q_requested;
  int32 aux_q_requested;
  int32 aux_q_end;
  int32 post_expand_aux_q_end;  // used for double buffering
  // Some tokens in the same frame share the same token.next_state
  // main_q_n_extra_prev_tokens is the count of those tokens
  int32 main_q_n_extra_prev_tokens;
  // Number of tokens created during the emitting stage
  int32 main_q_n_emitting_tokens;
  // Depending on the value of the parameter "max_tokens_per_frame"
  // we can end up with an overflow when generating the tokens for a frame
  // We try to prevent this from happening using an adaptive beam
  // If an overflow happens, then the kernels no longer insert any data into
  // the queues and set overflow flag to true.
  // queue length.
  // Even if that flag is set, we can continue the execution (quality
  // of the output can be lowered)
  // We use that flag to display a warning to the user
  int32 q_overflow;
  // ExpandArcs reads the tokens in the index range [main_q_local_offset, end[
  int32 main_q_local_offset;
  // We transfer the tokens back to the host at the end of each frame.
  // Which means that tokens at a frame  n > 0 have an offset compared to to
  // those
  // in frame n-1. main_q_global_offset is the overall offset of the current
  // main_q,
  // since frame 0
  // It is used to set the prev_token index.
  int32 main_q_global_offset;
  // Same thing, but for main_q_n_extra_prev_tokens (those are also transfered
  // back to host)
  int32 main_q_extra_prev_tokens_global_offset;
  // Minimum token for that frame
  IntegerCostType min_int_cost;
  IntegerCostType int_relative_cost;
  // Current beam. Can be different from default_beam,
  // because of the AdaptiveBeam process, or because of
  // ApplyMaxActiveAndReduceBeam
  IntegerCostType int_beam;
  // Adaptive beam. The validity says until which index this adaptive beam is
  // valid.
  // After that index, we need to lower the adaptive beam
  int2 adaptive_int_beam_with_validity_index;
  // min_cost + beam
  IntegerCostType int_cutoff;
  // The histogram for max_active will be computed between min_histo_cost
  // and max_histo_cost. Set for each frame after emitting stage
  CostType min_histo_cost;
  CostType max_histo_cost;
  CostType histo_bin_width;
  bool compute_max_active;
  // offsets used by concatenate_lanes_data_kernel
  int32 main_q_end_lane_offset;
  int32 main_q_n_emitting_tokens_lane_offset;
  int32 main_q_n_extra_prev_tokens_lane_offset;

  // --- Only valid after calling GetBestCost
  // min_cost and its arg. Can be different than min_cost, because we may
  // include final costs
  int2 min_int_cost_and_arg;
  // Number of final tokens with cost < best + lattice_beam
  int32 n_within_lattice_beam;
  int32 has_reached_final;  // if there's at least one final token in the queue
  int32 prev_arg_min_int_cost;
};

// Channel counters
// Their job is to save the state of a channel, when this channel is idle
// The channel counters are loaded into the lane counters during the context
// switches
struct ChannelCounters {
  // All the following values are just saved values from LaneCounters
  // from the latest context-switch
  int2 prev_main_q_narcs_and_end;
  int32 prev_main_q_n_extra_prev_tokens;
  int32 prev_main_q_global_offset;
  int32 prev_main_q_extra_prev_tokens_global_offset;
  CostType prev_beam;

  // Only valid after calling GetBestCost
  // different than min_int_cost : we include the "final" cost
  int2 min_int_cost_and_arg_with_final;
  int2 min_int_cost_and_arg_without_final;
};

class CudaDecoderException : public std::exception {
 public:
  CudaDecoderException(const char *str_, const char *file_, int line_,
                       const bool recoverable_)
      : str(str_),
        file(file_),
        line(line_),
        buffer(std::string(file) + ":" + std::to_string(line) + " :" +
               std::string(str)),
        recoverable(recoverable_) {}
  const char *what() const throw() { return buffer.c_str(); }

  const char *str;
  const char *file;
  const int line;
  const std::string buffer;
  const bool recoverable;
};

// Used to store the index in the GPU hashmap of that FST state
// The hashmap is only generated with the final main queue (post max_active_) of
// each frame
// Also stores the information or whether or not the owner of that object is the
// representative of this FSTState
typedef int32 FSTStateHashIndex;

// 1:1 Conversion float <---> sortable int
// We convert floats to sortable ints in order
// to use native atomics operation
// Those are the host version, used when we transfer an int from the device
// and we want to convert it to a float
// (it was created on device by floatToOrderedInt, we'll use
// orderedIntToFloatHost on host to convert it back to a float)
__inline__ int32 floatToOrderedIntHost(float floatVal) {
  int32 intVal;
  // Should be optimized away by compiler
  memcpy(&intVal, &floatVal, sizeof(float));
  return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}

__inline__ float orderedIntToFloatHost(int32 intVal) {
  intVal = (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
  float floatVal;
  // Should be optimized away by compiler
  memcpy(&floatVal, &intVal, sizeof(float));
  return floatVal;
}

// Hashmap value. Used when computing the hashmap in PostProcessingMainQueue
struct __align__(16) HashmapValueT {
  // Map key : fst state
  int32 key;
  // Number of tokens associated to that state
  int32 count;
  // minimum cost for that state + argmin
  unsigned long long min_and_argmin_int_cost_u64;
};

enum OVERFLOW_TYPE {
  OVERFLOW_NONE = 0,
  OVERFLOW_MAIN_Q = 1,
  OVERFLOW_AUX_Q = 2
};

enum QUEUE_ID { MAIN_Q = 0, AUX_Q = 1 };

// Used internally to generate partial paths
struct PartialPathArc {
  int32 token_idx;
  int32 arc_idx;
  int32 substring_end;
  int32 olabel;

  PartialPathArc(int32 _token_idx = -1, int32 _arc_idx = -1,
                 int32 _substring_end = -1)
      : token_idx(_token_idx),
        arc_idx(_arc_idx),
        substring_end(_substring_end),
        olabel(-1) {}
};

// Partial hypothesis formatted and meant to be used by user
struct PartialHypothesis {
  std::string out_str;

  void clear() { out_str.clear(); }
};

}  // end namespace cuda_decoder
}  // end namespace kaldi

#endif  // KALDI_CUDA_DECODER_CUDA_DECODER_UTILS_H_
