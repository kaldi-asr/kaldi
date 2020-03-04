// online2/online-nnet2-feature-pipeline.cc

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

#include "online2/online-nnet2-feature-pipeline.h"
#include "transform/cmvn.h"

namespace kaldi {

OnlineNnet2FeaturePipelineInfo::OnlineNnet2FeaturePipelineInfo(
    const OnlineNnet2FeaturePipelineConfig &config):
    silence_weighting_config(config.silence_weighting_config) {
  if (config.feature_type == "mfcc" || config.feature_type == "plp" ||
      config.feature_type == "fbank") {
    feature_type = config.feature_type;
  } else {
    KALDI_ERR << "Invalid feature type: " << config.feature_type << ". "
              << "Supported feature types: mfcc, plp, fbank.";
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

  if (config.online_pitch_config != "") {
    ReadConfigsFromFile(config.online_pitch_config,
                        &pitch_opts,
                        &pitch_process_opts);
    if (!add_pitch)
      KALDI_WARN << "--online-pitch-config option has no effect "
                 << "since you did not supply --add-pitch option.";
  }  // else use the defaults.

  use_cmvn = (config.cmvn_config != "");
  if (use_cmvn) {
    ReadConfigFromFile(config.cmvn_config, &cmvn_opts);
    global_cmvn_stats_rxfilename = config.global_cmvn_stats_rxfilename;
    if (global_cmvn_stats_rxfilename == "")
      KALDI_ERR << "--global-cmvn-stats option is required "
                << " when --cmvn-config is specified.";
  }

  if (config.ivector_extraction_config != "") {
    use_ivectors = true;
    OnlineIvectorExtractionConfig ivector_extraction_opts;
    ReadConfigFromFile(config.ivector_extraction_config,
                       &ivector_extraction_opts);
    ivector_extractor_info.Init(ivector_extraction_opts);
  } else {
    use_ivectors = false;
  }
}


/// The main feature extraction pipeline is constructed in this constructor.
OnlineNnet2FeaturePipeline::OnlineNnet2FeaturePipeline(
    const OnlineNnet2FeaturePipelineInfo &info):
    info_(info), base_feature_(NULL),
    pitch_(NULL), pitch_feature_(NULL),
    cmvn_feature_(NULL),
    feature_plus_optional_pitch_(NULL),
    feature_plus_optional_cmvn_(NULL),
    ivector_feature_(NULL),
    nnet3_feature_(NULL),
    final_feature_(NULL) {

  if (info_.feature_type == "mfcc") {
    base_feature_ = new OnlineMfcc(info_.mfcc_opts);
  } else if (info_.feature_type == "plp") {
    base_feature_ = new OnlinePlp(info_.plp_opts);
  } else if (info_.feature_type == "fbank") {
    base_feature_ = new OnlineFbank(info_.fbank_opts);
  } else {
    KALDI_ERR << "Code error: invalid feature type " << info_.feature_type;
  }

  if (info_.add_pitch) {
    pitch_ = new OnlinePitchFeature(info_.pitch_opts);
    pitch_feature_ = new OnlineProcessPitch(info_.pitch_process_opts,
                                            pitch_);
    feature_plus_optional_pitch_ = new OnlineAppendFeature(base_feature_,
                                                           pitch_feature_);
  } else {
    feature_plus_optional_pitch_ = base_feature_;
  }

  if (info_.use_cmvn) {
    KALDI_ASSERT(info.global_cmvn_stats_rxfilename != "");
    ReadKaldiObject(info.global_cmvn_stats_rxfilename, &global_cmvn_stats_);
    OnlineCmvnState initial_state(global_cmvn_stats_);
    cmvn_feature_ = new OnlineCmvn(info_.cmvn_opts, initial_state,
        feature_plus_optional_pitch_);
    feature_plus_optional_cmvn_ = cmvn_feature_;
  } else {
    feature_plus_optional_cmvn_ = feature_plus_optional_pitch_;
  }

  if (info_.use_ivectors) {
    nnet3_feature_ = feature_plus_optional_cmvn_;
    // Note: the i-vector extractor OnlineIvectorFeature gets 'base_feautre_'
    // without cmvn (the online cmvn is applied inside the class)
    ivector_feature_ = new OnlineIvectorFeature(info_.ivector_extractor_info,
                                                base_feature_);
    final_feature_ = new OnlineAppendFeature(feature_plus_optional_cmvn_,
                                             ivector_feature_);
  } else {
    nnet3_feature_ = feature_plus_optional_cmvn_;
    final_feature_ = feature_plus_optional_cmvn_;
  }
  dim_ = final_feature_->Dim();
}
/// ^-^


int32 OnlineNnet2FeaturePipeline::Dim() const { return dim_; }

bool OnlineNnet2FeaturePipeline::IsLastFrame(int32 frame) const {
  return final_feature_->IsLastFrame(frame);
}

int32 OnlineNnet2FeaturePipeline::NumFramesReady() const {
  return final_feature_->NumFramesReady();
}

void OnlineNnet2FeaturePipeline::GetFrame(int32 frame,
                                          VectorBase<BaseFloat> *feat) {
  return final_feature_->GetFrame(frame, feat);
}

void OnlineNnet2FeaturePipeline::UpdateFrameWeights(
    const std::vector<std::pair<int32, BaseFloat> > &delta_weights) {
    IvectorFeature()->UpdateFrameWeights(delta_weights);
}

void OnlineNnet2FeaturePipeline::SetAdaptationState(
    const OnlineIvectorExtractorAdaptationState &adaptation_state) {
  if (info_.use_ivectors) {
    ivector_feature_->SetAdaptationState(adaptation_state);
  }
  // else silently do nothing, as there is nothing to do.
}

void OnlineNnet2FeaturePipeline::GetAdaptationState(
    OnlineIvectorExtractorAdaptationState *adaptation_state) const {
  if (info_.use_ivectors) {
    ivector_feature_->GetAdaptationState(adaptation_state);
  }
  // else silently do nothing, as there is nothing to do.
}

void OnlineNnet2FeaturePipeline::SetCmvnState(
    const OnlineCmvnState &cmvn_state) {
  if (NULL != cmvn_feature_)
    cmvn_feature_->SetState(cmvn_state);
}

void OnlineNnet2FeaturePipeline::GetCmvnState(
    OnlineCmvnState *cmvn_state) {
  if (NULL != cmvn_feature_) {
    int32 frame = cmvn_feature_->NumFramesReady() - 1;
    // the following call will crash if no frames are ready.
    cmvn_feature_->GetState(frame, cmvn_state);
  }
}


OnlineNnet2FeaturePipeline::~OnlineNnet2FeaturePipeline() {
  // Note: the delete command only deletes pointers that are non-NULL.  Not all
  // of the pointers below will be non-NULL.
  // Some of the online-feature pointers are just copies of other pointers,
  // and we do have to avoid deleting them in those cases.
  if (final_feature_ != feature_plus_optional_cmvn_)
    delete final_feature_;
  delete ivector_feature_;
  delete cmvn_feature_;
  if (feature_plus_optional_pitch_ != base_feature_)
    delete feature_plus_optional_pitch_;
  delete pitch_feature_;
  delete pitch_;
  delete base_feature_;
}

void OnlineNnet2FeaturePipeline::AcceptWaveform(
    BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &waveform) {
  base_feature_->AcceptWaveform(sampling_rate, waveform);
  if (pitch_)
    pitch_->AcceptWaveform(sampling_rate, waveform);
}

void OnlineNnet2FeaturePipeline::InputFinished() {
  base_feature_->InputFinished();
  if (pitch_)
    pitch_->InputFinished();
}

BaseFloat OnlineNnet2FeaturePipelineInfo::FrameShiftInSeconds() const {
  if (feature_type == "mfcc") {
    return mfcc_opts.frame_opts.frame_shift_ms / 1000.0f;
  } else if (feature_type == "fbank") {
    return fbank_opts.frame_opts.frame_shift_ms / 1000.0f;
  } else if (feature_type == "plp") {
    return plp_opts.frame_opts.frame_shift_ms / 1000.0f;
  } else {
    KALDI_ERR << "Unknown feature type " << feature_type;
    return 0.0;
  }
}

BaseFloat OnlineNnet2FeaturePipelineInfo::GetSamplingFrequency() {
  if (feature_type == "mfcc") {
    return mfcc_opts.frame_opts.samp_freq;
  } else if (feature_type == "plp") {
    return plp_opts.frame_opts.samp_freq;
  } else if (feature_type == "fbank") {
    return fbank_opts.frame_opts.samp_freq;
  } else {
    KALDI_ERR << "Unknown feature type " << feature_type;
  }
  return 0.0f; // avoiding a possible "return missing" warning
}

}  // namespace kaldi
