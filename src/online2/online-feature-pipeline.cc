// online2/online-feature-pipeline.cc

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

#include "online2/online-feature-pipeline.h"
#include "transform/cmvn.h"

namespace kaldi {


OnlineFeaturePipelineConfig::OnlineFeaturePipelineConfig(
    const OnlineFeaturePipelineCommandLineConfig &config) {
  if (config.mfcc_config != "") {
    ReadConfigFromFile(config.mfcc_config, &mfcc_opts);
    feature_type = "mfcc";
  } else if (config.plp_config != "") {
    ReadConfigFromFile(config.plp_config, &plp_opts);
    feature_type = "plp";
  } else {
    KALDI_ERR << "Either the --mfcc-config or --plp-config options "
              << "must be supplied.";
  }
  if (config.cmvn_config != "") {
    ReadConfigFromFile(config.cmvn_config, &cmvn_opts);
  } // else use the defaults.

  if (config.splice_config != "") {
    ReadConfigFromFile(config.splice_config, &splice_opts);
    splice_frames = true;
  } else {
    splice_frames = false;
  }
  if (config.delta_config != "") {
    ReadConfigFromFile(config.delta_config, &delta_opts);
    apply_deltas = true;
  } else {
    apply_deltas = false;
  }
  if (splice_frames && apply_deltas)
    KALDI_ERR << "You cannot supply both the --splice-config "
              << "and --delta-config options";
  lda_rxfilename = config.lda_rxfilename;
  global_cmvn_stats_rxfilename = config.global_cmvn_stats_rxfilename;
  if (global_cmvn_stats_rxfilename == "")
    KALDI_ERR << "--global-cmvn-stats option is required.";
}


OnlineFeaturePipeline::OnlineFeaturePipeline(
    const OnlineFeaturePipelineConfig &config,
    const Matrix<BaseFloat> &lda_mat,
    const Matrix<BaseFloat> &global_cmvn_stats):
    config_(config), lda_mat_(lda_mat), global_cmvn_stats_(global_cmvn_stats) {
  Init();
}


OnlineFeaturePipeline::OnlineFeaturePipeline(
    const OnlineFeaturePipelineConfig &config):
    config_(config) {
  if (config.lda_rxfilename != "")
    ReadKaldiObject(config.lda_rxfilename, &lda_mat_);    
  if (config.global_cmvn_stats_rxfilename != "")
    ReadKaldiObject(config.global_cmvn_stats_rxfilename,
                    &global_cmvn_stats_);    
  Init();
}

OnlineFeaturePipeline* OnlineFeaturePipeline::New() const {
  return new OnlineFeaturePipeline(config_, lda_mat_,
                                   global_cmvn_stats_);
}

OnlineFeatureInterface* OnlineFeaturePipeline::UnadaptedFeature() const {
  if (lda_) return lda_;
  else if (splice_or_delta_) return splice_or_delta_;
  else {
    KALDI_ASSERT(cmvn_ != NULL);
    return cmvn_;
  }
}

OnlineFeatureInterface* OnlineFeaturePipeline::AdaptedFeature() const {
  if (fmllr_) return fmllr_;
  else return UnadaptedFeature();
}


void OnlineFeaturePipeline::SetCmvnState(const OnlineCmvnState &cmvn_state) {
  cmvn_->SetState(cmvn_state);
}

void OnlineFeaturePipeline::GetCmvnState(OnlineCmvnState *cmvn_state) {
  int32 frame = cmvn_->NumFramesReady() - 1;
  // the following call will crash if no frames are ready.
  cmvn_->GetState(frame, cmvn_state);
}


// Init() is to be called from the constructor; it assumes the pointer
// members are all uninitialized but config_ and lda_mat_ are
// initialized.
void OnlineFeaturePipeline::Init() {
  if (config_.feature_type == "mfcc") {
    base_feature_ = new OnlineMfcc(config_.mfcc_opts);
  } else if (config_.feature_type == "plp") {
    base_feature_ = new OnlinePlp(config_.plp_opts);
  } else {
    KALDI_ERR << "Code error: invalid feature type " << config_.feature_type;
  }

  {
    KALDI_ASSERT(global_cmvn_stats_.NumRows() != 0);
    Matrix<double> global_cmvn_stats_dbl(global_cmvn_stats_);
    OnlineCmvnState initial_state(global_cmvn_stats_dbl);
    cmvn_ = new OnlineCmvn(config_.cmvn_opts, initial_state, base_feature_);
  }

  if (config_.splice_frames && config_.apply_deltas) {
    KALDI_ERR << "You cannot supply both the --delta-config and "
              << "--splice-config options";
  } else if (config_.splice_frames) {
    splice_or_delta_ = new OnlineSpliceFrames(config_.splice_opts,
                                              cmvn_);
  } else if (config_.apply_deltas) {
    splice_or_delta_ = new OnlineDeltaFeature(config_.delta_opts,
                                              cmvn_);
  } else {
    splice_or_delta_ = NULL;
  }
  
  if (lda_mat_.NumRows() != 0) {
    lda_ = new OnlineTransform(lda_mat_,
                               (splice_or_delta_ != NULL ?
                                splice_or_delta_ : cmvn_));
  }

  fmllr_ = NULL; // This will be set up if the user calls SetTransform().
}

void OnlineFeaturePipeline::SetTransform(
    const MatrixBase<BaseFloat> &transform) {
  if (fmllr_ != NULL) { // we already had a transform;  delete this
    // object.
    delete fmllr_;
    fmllr_ = NULL;
  }
  if (transform.NumRows() != 0) {
    OnlineFeatureInterface *feat = UnadaptedFeature();
    fmllr_ = new OnlineTransform(transform, feat);
  }
}


void OnlineFeaturePipeline::FreezeCmvn() {
  cmvn_->Freeze(cmvn_->NumFramesReady() - 1);  
}

int32 OnlineFeaturePipeline::Dim() const {
  return AdaptedFeature()->Dim();
}
bool OnlineFeaturePipeline::IsLastFrame(int32 frame) const {
  return AdaptedFeature()->IsLastFrame(frame);
}
int32 OnlineFeaturePipeline::NumFramesReady() const {
  return AdaptedFeature()->NumFramesReady();
}

void OnlineFeaturePipeline::GetFrame(int32 frame,
                                        VectorBase<BaseFloat> *feat) {
  AdaptedFeature()->GetFrame(frame, feat);
}

OnlineFeaturePipeline::~OnlineFeaturePipeline() {
  // Note: the delete command only deletes pointers that are non-NULL.  Not all
  // of the pointers below will be non-NULL.
  delete fmllr_;
  delete lda_;
  delete splice_or_delta_;
  delete cmvn_;
  delete base_feature_;
}

void OnlineFeaturePipeline::AcceptWaveform(
    BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &waveform) {
  base_feature_->AcceptWaveform(sampling_rate, waveform);
}

void OnlineFeaturePipeline::InputFinished() {
  base_feature_->InputFinished();
}


}  // namespace kaldi
