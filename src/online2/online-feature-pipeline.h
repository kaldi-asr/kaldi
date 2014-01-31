// online2/online-feature-pipeline.h

// Copyright 2013   Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_ONLINE2_ONLINE_FEATURE_PIPELINE_H_
#define KALDI_ONLINE2_ONLINE_FEATURE_PIPELINE_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "online2/online-feature.h"

namespace kaldi {
/// @addtogroup  onlinefeat OnlineFeatureExtraction
/// @{



struct OnlineFeaturePipelineConfig {
  std::string mfcc_config;
  std::string plp_config;
  // later will have:
  // std::string pitch_config;
  std::string cmvn_online_config;
  std::string global_cmvn_stats_rxfilename;
  std::string delta_config;
  std::string lda_rxfilename;
  
  OnlineFeaturePipelineConfig();

  void Register(OptionsItf *po) {

  }
};


class OnlineFeaturePipeline: public OnlineFeatureInterface {
 public:
  OnlineFeaturePipeline(OnlineFeaturePipelineConfig &cfg);

  /// Member functions from OnlineFeatureInterface:
  virtual int32 Dim() const;
  virtual bool IsLastFrame(int32 frame) const;
  virtual int32 NumFramesReady() const;
  virtual void GetFeatures(int32 frame, VectorBase<BaseFloat> *feat);
  

  /// Returns the expected sample frequency of the wave input.
  BaseFloat SampFreq() const;

  /// Completely reset the state of the object.
  void Reset();
  
  void ResetCmvn(); // could make this take arguments?
  void FreezeCmvn(); // stop it from moving further (do this when you start using
                     // fMLLR)

  /// Accept more data to process (won't actually process it, will
  /// just copy it). 
  void AcceptWaveform(BaseFloat sampling_rate,
                      const VectorBase<BaseFloat> &waveform);
  
  // This object is used to set the fMLLR transform.  Call it with
  // the empty matrix if you want to stop it using any transform.
  void SetTransform(const MatrixBase<BaseFloat> &transform);

  // Returns a reference to the currently used fMLLR transform, or
  // the empty matrix if none is being used.  Can be used to
  // tell whether the pipeline has a transform currently.
  const MatrixBase<BaseFloat> &GetTransform() const;
  
  // returns a copy of *this.  Only applicable for a "fresh" pipeline,
  // that has not had any features added to it. 
  OnlineFeaturePipeline *Copy() const;
  
};




/// @} End of "addtogroup onlinefeat"
}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_FEATURE_PIPELINE_H_
