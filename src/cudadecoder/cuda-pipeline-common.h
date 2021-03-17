// cudadecoder/cuda-pipeline-common.h
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

#if HAVE_CUDA == 1

#ifndef KALDI_CUDA_DECODER_CUDA_PIPELINE_COMMON_
#define KALDI_CUDA_DECODER_CUDA_PIPELINE_COMMON_

// Initial size of the dynamic pinned host array
#define KALDI_CUDA_DECODER_AUDIO_HOST_DEVICE_BUFFER_SIZE 16000 * 50

#include "base/kaldi-utils.h"
#include "cudamatrix/cu-device.h"
#include "lat/lattice-functions.h"
#include "util/stl-utils.h"

namespace kaldi {
namespace cuda_decoder {

// Number of segments of a given length and shift in an utterance of total
// length nsamples
inline int NumberOfSegments(int nsamples, int seg_length, int seg_shift) {
  KALDI_ASSERT(seg_shift > 0);
  KALDI_ASSERT(seg_length >= seg_shift);
  int r = seg_length - seg_shift;
  if (nsamples <= seg_length) return 1;
  int nsegments = ((nsamples - r) + seg_shift - 1) / seg_shift;
  return nsegments;
}

// Segmentation config struct, used in cuda pipelines
struct CudaPipelineSegmentationConfig {
  double segment_length_s;
  double segment_overlap_s;
  double min_segment_length_s;

  CudaPipelineSegmentationConfig()
      : segment_length_s(20), segment_overlap_s(1), min_segment_length_s(1) {}
  void Register(OptionsItf *po) {
    po->Register("segment-length", &segment_length_s, "Segment length (s)");
    po->Register("segment-overlap", &segment_overlap_s,
                 "Overlap between segments (s)");
    po->Register("min-segment-length", &min_segment_length_s,
                 "Min segment length (s, >=1)");
  }

  void Check() const {
    if (min_segment_length_s < 0.5)
      KALDI_ERR << "Min segment length must be at least 0.5 second";
    if (segment_overlap_s > segment_length_s)
      KALDI_ERR << "The segments overlap cannot be larger than segment length";
  }
};

struct CTMResult {
  std::vector<BaseFloat> conf;
  std::vector<int32> words;
  std::vector<std::pair<BaseFloat, BaseFloat> > times_seconds;
};

// Struct to provide a result back to the user
class CudaPipelineResult {
  int result_type_;
  CompactLattice clat_;
  CTMResult ctm_result_;
  BaseFloat offset_seconds_;

 public:
  static constexpr int RESULT_TYPE_LATTICE = 1;
  static constexpr int RESULT_TYPE_CTM = 2;

  CudaPipelineResult() : result_type_(0), offset_seconds_(0) {}

  int32 GetResultType() const { return result_type_; }

  bool HasValidResult() const { return result_type_; }

  void SetLatticeResult(CompactLattice &&clat) {
    result_type_ |= RESULT_TYPE_LATTICE;
    // We can switch to std::forward if there's a use for lvalues
    clat_ = std::move(clat);
  }

  void SetCTMResult(CTMResult &&ctm) {
    result_type_ |= RESULT_TYPE_CTM;
    ctm_result_ = std::move(ctm);
  }

  const CompactLattice &GetLatticeResult() const {
    KALDI_ASSERT("Lattice result was not requested" &&
                 result_type_ & RESULT_TYPE_LATTICE);
    return clat_;
  }

  const CTMResult &GetCTMResult() const {
    KALDI_ASSERT("CTM result was not requested" &&
                 result_type_ & RESULT_TYPE_CTM);
    return ctm_result_;
  }

  void SetTimeOffsetSeconds(BaseFloat offset_seconds) {
    KALDI_ASSERT(offset_seconds >= 0);
    offset_seconds_ = offset_seconds;
  }

  BaseFloat GetTimeOffsetSeconds() const { return offset_seconds_; }
};

struct SegmentedLatticeCallbackParams {
  std::vector<CudaPipelineResult> results;
};

typedef std::function<void(SegmentedLatticeCallbackParams &params)>
    SegmentedResultsCallback;
typedef std::function<void(CompactLattice &)> LatticeCallback;

struct HostDeviceVector {
  cudaEvent_t evt;
  BaseFloat *h_data;
  BaseFloat *d_data;
  size_t size;

  HostDeviceVector(
      const size_t new_size = KALDI_CUDA_DECODER_AUDIO_HOST_DEVICE_BUFFER_SIZE)
      : h_data(NULL), d_data(NULL), size(new_size) {
    cudaEventCreate(&evt);
    Reallocate(new_size);
  }

  virtual ~HostDeviceVector() {
    Deallocate();
    cudaEventDestroy(evt);
  }

  void Reallocate(const size_t new_size) {
    KALDI_ASSERT(new_size > 0);
    Deallocate();

    cudaError_t cuResult = cudaSuccess;
    cuResult = cudaMalloc(&d_data, new_size * sizeof(BaseFloat));
    if (cuResult != cudaSuccess) {
      KALDI_ERR << "cudaMalloc() failed with error: "
                << cudaGetErrorString(cuResult);
    }
    KALDI_ASSERT(d_data != NULL);

    cuResult = cudaMallocHost(&h_data, new_size * sizeof(BaseFloat));
    if (cuResult != cudaSuccess) {
      KALDI_ERR << "cudaMallocHost() failed with error: "
                << cudaGetErrorString(cuResult);
    }
    KALDI_ASSERT(h_data != NULL);

    size = new_size;
  }
  void Deallocate() {
    if (d_data) {
      cudaFree(d_data);
      d_data = NULL;
    }
    if (h_data) {
      cudaFreeHost(h_data);
      h_data = NULL;
    }
  }
};
}  // namespace cuda_decoder

}  // namespace kaldi

#endif  // KALDI_CUDA_DECODER_CUDA_PIPELINE_COMMON_
#endif  // HAVE_CUDA
