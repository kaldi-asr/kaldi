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
    : info_(config), mfcc(NULL), ivector(NULL) {
  if (info_.feature_type == "mfcc") {
    mfcc = new CudaMfcc(info_.mfcc_opts);
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
  if (mfcc != NULL) delete mfcc;
  if (ivector != NULL) delete ivector;
}

void OnlineCudaFeaturePipeline::ComputeFeatures(
    const CuVectorBase<BaseFloat> &cu_wave, BaseFloat sample_freq,
    CuMatrix<BaseFloat> *input_features,
    CuVector<BaseFloat> *ivector_features) {
  if (info_.feature_type == "mfcc") {
    // MFCC
    float vtln_warp = 1.0;
    mfcc->ComputeFeatures(cu_wave, sample_freq, vtln_warp, input_features);
  } else {
    KALDI_ASSERT(false);
  }

  // Ivector
  if (info_.use_ivectors && ivector_features != NULL) {
    ivector->GetIvector(*input_features, ivector_features);
  } else {
    KALDI_ASSERT(false);
  }
}

}  // namespace kaldi
