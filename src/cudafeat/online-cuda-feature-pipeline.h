// cudafeat/online-cuda-feature-pipeline.h

// Copyright 2013-2014   Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_CUDAFEAT_ONLINE_CUDA_FEATURE_PIPELINE_H_
#define KALDI_CUDAFEAT_ONLINE_CUDA_FEATURE_PIPELINE_H_

#include <deque>
#include <string>
#include <vector>

#include "base/kaldi-error.h"
#include "cudafeat/feature-spectral-cuda.h"
#include "cudafeat/online-ivector-feature-cuda.h"
#include "matrix/matrix-lib.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "util/common-utils.h"

namespace kaldi {

class OnlineCudaFeaturePipeline {
 public:
  explicit OnlineCudaFeaturePipeline(
      const OnlineNnet2FeaturePipelineConfig &config);

  void ComputeFeatures(const CuVectorBase<BaseFloat> &cu_wave,
                       BaseFloat sample_freq,
                       CuMatrix<BaseFloat> *input_features,
                       CuVector<BaseFloat> *ivector_features);

  ~OnlineCudaFeaturePipeline();

 private:
  OnlineNnet2FeaturePipelineInfo info_;
  CudaSpectralFeatures *spectral_feat;
  CudaOnlineCmvn *cmvn;
  IvectorExtractorFastCuda *ivector;
  Matrix<double> global_cmvn_stats;
};
}  // namespace kaldi

#endif  // KALDI_CUDAFEAT_ONLINE_CUDA_FEATURE_EXTRACTOR_H_
