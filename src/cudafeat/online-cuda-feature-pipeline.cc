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
    const OnlineNnet2FeaturePipelineConfig &config)
    : info_(config), spectral_feat(NULL), ivector(NULL) {
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
    KALDI_ASSERT(info_.global_cmvn_stats_rxfilename != "");
    ReadKaldiObject(info_.global_cmvn_stats_rxfilename, &global_cmvn_stats);
    OnlineCmvnState cmvn_state(global_cmvn_stats);
    CudaOnlineCmvnState cu_cmvn_state(cmvn_state);
    cmvn = new CudaOnlineCmvn(info_.cmvn_opts, cu_cmvn_state);
  } 

  if (info_.use_ivectors) {
    OnlineIvectorExtractionConfig ivector_extraction_opts;
    ReadConfigFromFile(config.ivector_extraction_config,
                       &ivector_extraction_opts);
    info_.ivector_extractor_info.Init(ivector_extraction_opts);

    // Only these ivector options are currently supported
    ivector_extraction_opts.use_most_recent_ivector = true;
    ivector_extraction_opts.greedy_ivector_extractor = true;

    ivector = new IvectorExtractorFastCuda(ivector_extraction_opts);
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
