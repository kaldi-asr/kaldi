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
  if (config.feature_type == "mfcc" || config.feature_type == "plp" ||
      config.feature_type == "fbank") {
    feature_type = config.feature_type;
  } else {
    KALDI_ERR << "Invalid feature type: " << config.feature_type << ". "
              << "Supported feature types: mfcc, plp.";
  }

  if (config.mfcc_config != "") {
    ReadConfigFromFile(config.mfcc_config, &mfcc_opts);
    if (feature_type != "mfcc")
      KALDI_WARN << "--mfcc-config option has no effect "
                 << "since feature type is set to " << feature_type << ".";
  }  // else use the defaults.

  if (config.plp_config != "") {
    ReadConfigFromFile(config.plp_config, &plp_opts);
    if (feature_type != "plp")
      KALDI_WARN << "--plp-config option has no effect "
                 << "since feature type is set to " << feature_type << ".";
  }  // else use the defaults.

  if (config.fbank_config != "") {
    ReadConfigFromFile(config.fbank_config, &fbank_opts);
    if (feature_type != "fbank")
      KALDI_WARN << "--fbank-config option has no effect "
                 << "since feature type is set to " << feature_type << ".";
  }  // else use the defaults.

  add_pitch = config.add_pitch;
  if (config.pitch_config != "") {
    ReadConfigFromFile(config.pitch_config, &pitch_opts);
    if (!add_pitch)
      KALDI_WARN << "--pitch-config option has no effect "
                 << "since you did not supply --add-pitch option.";
  }  // else use the defaults.

  if (config.pitch_process_config != "") {
    ReadConfigFromFile(config.pitch_process_config, &pitch_process_opts);
    if (!add_pitch)
      KALDI_WARN << "--pitch-process-config option has no effect "
                 << "since you did not supply --add-pitch option.";
  }  // else use the defaults.

  if (config.cmvn_config != "") {
    ReadConfigFromFile(config.cmvn_config, &cmvn_opts);
  }  // else use the defaults.

  global_cmvn_stats_rxfilename = config.global_cmvn_stats_rxfilename;
  if (global_cmvn_stats_rxfilename == "")
    KALDI_ERR << "--global-cmvn-stats option is required.";

  add_deltas = config.add_deltas;
  if (config.delta_config != "") {
    ReadConfigFromFile(config.delta_config, &delta_opts);
    if (!add_deltas)
      KALDI_WARN << "--delta-config option has no effect "
                 << "since you did not supply --add-deltas option.";
  }  // else use the defaults.

  splice_feats = config.splice_feats;
  if (config.splice_config != "") {
    ReadConfigFromFile(config.splice_config, &splice_opts);
    if (!splice_feats)
      KALDI_WARN << "--splice-config option has no effect "
                 << "since you did not supply --splice-feats option.";
  }  // else use the defaults.

  if (config.add_deltas && config.splice_feats)
    KALDI_ERR << "You cannot supply both --add-deltas "
              << "and --splice-feats options";

  lda_rxfilename = config.lda_rxfilename;
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
    KALDI_ASSERT(feature_ != NULL);
    return feature_;
  }
}

OnlineFeatureInterface* OnlineFeaturePipeline::AdaptedFeature() const {
  if (fmllr_) return fmllr_;
  else
    return UnadaptedFeature();
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
  } else if (config_.feature_type == "fbank") {
    base_feature_ = new OnlineFbank(config_.fbank_opts);
  } else {
    KALDI_ERR << "Code error: invalid feature type " << config_.feature_type;
  }

  {
    KALDI_ASSERT(global_cmvn_stats_.NumRows() != 0);
    if (config_.add_pitch) {
      int32 global_dim = global_cmvn_stats_.NumCols() - 1;
      int32 dim = base_feature_->Dim();
      KALDI_ASSERT(global_dim >= dim);
      if (global_dim > dim) {
        Matrix<BaseFloat> last_col(global_cmvn_stats_.ColRange(global_dim, 1));
        global_cmvn_stats_.Resize(global_cmvn_stats_.NumRows(), dim + 1,
                                  kCopyData);
        global_cmvn_stats_.ColRange(dim, 1).CopyFromMat(last_col);
      }
    }
    Matrix<double> global_cmvn_stats_dbl(global_cmvn_stats_);
    OnlineCmvnState initial_state(global_cmvn_stats_dbl);
    cmvn_ = new OnlineCmvn(config_.cmvn_opts, initial_state, base_feature_);
  }

  if (config_.add_pitch) {
    pitch_ = new OnlinePitchFeature(config_.pitch_opts);
    pitch_feature_ = new OnlineProcessPitch(config_.pitch_process_opts,
                                            pitch_);
    feature_ = new OnlineAppendFeature(cmvn_, pitch_feature_);
  } else {
    pitch_ = NULL;
    pitch_feature_ = NULL;
    feature_ = cmvn_;
  }

  if (config_.splice_feats && config_.add_deltas) {
    KALDI_ERR << "You cannot supply both --add-deltas and "
              << "--splice-feats options.";
  } else if (config_.splice_feats) {
    splice_or_delta_ = new OnlineSpliceFrames(config_.splice_opts,
                                              feature_);
  } else if (config_.add_deltas) {
    splice_or_delta_ = new OnlineDeltaFeature(config_.delta_opts,
                                              feature_);
  } else {
    splice_or_delta_ = NULL;
  }

  if (lda_mat_.NumRows() != 0) {
    lda_ = new OnlineTransform(lda_mat_,
                               (splice_or_delta_ != NULL ?
                                splice_or_delta_ : feature_));
  } else {
    lda_ = NULL;
  }

  fmllr_ = NULL;  // This will be set up if the user calls SetTransform().
}

void OnlineFeaturePipeline::SetTransform(
    const MatrixBase<BaseFloat> &transform) {
  if (fmllr_ != NULL) {  // we already had a transform;  delete this
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
  // Guard against double deleting the cmvn_ ptr
  if (pitch_feature_) {
    delete feature_;  // equal to cmvn_ if pitch feats are not appended
    delete pitch_feature_;
    delete pitch_;
  }
  delete cmvn_;
  delete base_feature_;
}

void OnlineFeaturePipeline::AcceptWaveform(
    BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &waveform) {
  base_feature_->AcceptWaveform(sampling_rate, waveform);
  if (pitch_)
    pitch_->AcceptWaveform(sampling_rate, waveform);
}

void OnlineFeaturePipeline::InputFinished() {
  base_feature_->InputFinished();
  if (pitch_)
    pitch_->InputFinished();
}

BaseFloat OnlineFeaturePipelineConfig::FrameShiftInSeconds() const {
  if (feature_type == "mfcc") {
    return mfcc_opts.frame_opts.frame_shift_ms / 1000.0f;
  } else if (feature_type == "plp") {
    return plp_opts.frame_opts.frame_shift_ms / 1000.0f;
  } else {
    KALDI_ERR << "Unknown feature type " << feature_type;
    return 0.0;
  }
}

void OnlineFeaturePipeline::GetAsMatrix(Matrix<BaseFloat> *feats) {
  if (pitch_) {
    feats->Resize(NumFramesReady(), pitch_feature_->Dim());
    for (int32 i = 0; i < NumFramesReady(); i++) {
      SubVector<BaseFloat> row(*feats, i);
      pitch_feature_->GetFrame(i, &row);
    }
  }
}

}  // namespace kaldi
