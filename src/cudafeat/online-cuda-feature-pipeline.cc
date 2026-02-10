// cudafeat/online-cuda-feature-pipleine.cc

// Copyright    2013  Johns Hopkins University (author: Daniel Povey)

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

#include "cudafeat/online-cuda-feature-pipeline.h"

namespace kaldi {

OnlineCudaFeaturePipeline::OnlineCudaFeaturePipeline(
    const OnlineNnet2FeaturePipelineInfo &info)
    : info_(info), spectral_feat(NULL), ivector(NULL) {
  spectral_feat = NULL;
  cmvn = NULL;
  ivector = NULL;
  if (info_.feature_type == "mfcc") {
    spectral_feat = new CudaSpectralFeatures(info_.mfcc_opts);
  }
  if (info_.feature_type == "fbank") {
    spectral_feat = new CudaSpectralFeatures(info_.fbank_opts);
  }

  if (info_.use_cmvn) {
    if (info_.global_cmvn_stats.NumCols() == 0) {
      KALDI_ERR << "global_cmvn_stats for OnlineCmvn must be non-empty.";
    }
    OnlineCmvnState cmvn_state(info_.global_cmvn_stats);
    CudaOnlineCmvnState cu_cmvn_state(cmvn_state);
    cmvn = new CudaOnlineCmvn(info_.cmvn_opts, cu_cmvn_state);
  }

  if (info_.use_ivectors) {
    ivector = new IvectorExtractorFastCuda(info_.ivector_extractor_info);
  }
}

OnlineCudaFeaturePipeline::~OnlineCudaFeaturePipeline() {
  if (spectral_feat != NULL) delete spectral_feat;
  if (cmvn != NULL) delete cmvn;
  if (ivector != NULL) delete ivector;
}

void OnlineCudaFeaturePipeline::ComputeFeatures(
    const CuVectorBase<BaseFloat> &cu_wave, BaseFloat sample_freq,
    CuMatrix<BaseFloat> *input_features,
    CuVector<BaseFloat> *ivector_features) {
  if (info_.feature_type == "mfcc" || info_.feature_type == "fbank") {
    // Fbank called via the MFCC codepath
    // MFCC
    float vtln_warp = 1.0;
    spectral_feat->ComputeFeatures(cu_wave, sample_freq, vtln_warp, input_features);
  } else {
    KALDI_ASSERT(false);
  }

  if (info_.use_cmvn) {
    cmvn->ComputeFeatures(*input_features, input_features);
  }

  // Ivector
  if (info_.use_ivectors && ivector_features != NULL) {
    ivector->GetIvector(*input_features, ivector_features);
  }
}

}  // namespace kaldi
