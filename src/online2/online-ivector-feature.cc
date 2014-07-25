// online2/online-ivector-feature.cc

// Copyright 2014  Daniel Povey

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

#include "online2/online-ivector-feature.h"

namespace kaldi {

OnlineIvectorExtractionInfo::OnlineIvectorExtractionInfo(
    const OnlineIvectorExtractionConfig &config) {
  Init(config);
}

void OnlineIvectorExtractionInfo::Init(
    const OnlineIvectorExtractionConfig &config) {
  ivector_period = config.ivector_period;
  num_gselect = config.num_gselect;
  min_post = config.min_post;
  posterior_scale = config.posterior_scale;
  use_most_recent_ivector = config.use_most_recent_ivector;
  max_remembered_frames = config.max_remembered_frames;
  
  std::string note = "(note: this may be needed "
      "in the file supplied to --ivector-extractor-config)";
  if (config.lda_mat_rxfilename == "")
    KALDI_ERR << "--lda-matrix option must be set " << note;
  ReadKaldiObject(config.lda_mat_rxfilename, &lda_mat);
  if (config.global_cmvn_stats_rxfilename == "")
    KALDI_ERR << "--global-cmvn-stats option must be set " << note;
  ReadKaldiObject(config.global_cmvn_stats_rxfilename, &global_cmvn_stats);
  if (config.cmvn_config_rxfilename == "")
    KALDI_ERR << "--cmvn-config option must be set " << note;
  ReadConfigFromFile(config.cmvn_config_rxfilename, &cmvn_opts);
  if (config.splice_config_rxfilename == "")
    KALDI_ERR << "--splice-config option must be set " << note;
  ReadConfigFromFile(config.splice_config_rxfilename, &splice_opts);
  if (config.diag_ubm_rxfilename == "")
    KALDI_ERR << "--diag-ubm option must be set " << note;
  ReadKaldiObject(config.diag_ubm_rxfilename, &diag_ubm);
  if (config.ivector_extractor_rxfilename == "")
    KALDI_ERR << "--ivector-extractor option must be set " << note;
  ReadKaldiObject(config.ivector_extractor_rxfilename, &extractor);
  this->Check();
}


void OnlineIvectorExtractionInfo::Check() const {
  KALDI_ASSERT(global_cmvn_stats.NumRows() == 2);
  int32 base_feat_dim = global_cmvn_stats.NumCols() - 1,
      num_splice = splice_opts.left_context + 1 + splice_opts.right_context,
      spliced_input_dim = base_feat_dim * num_splice;
  
  KALDI_ASSERT(lda_mat.NumCols() == spliced_input_dim ||
               lda_mat.NumCols() == spliced_input_dim + 1);
  KALDI_ASSERT(lda_mat.NumRows() == diag_ubm.Dim());
  KALDI_ASSERT(diag_ubm.Dim() == extractor.FeatDim());
  KALDI_ASSERT(ivector_period > 0);
  KALDI_ASSERT(num_gselect > 0);
  KALDI_ASSERT(min_post < 0.5);
  // posterior scale more than one does not really make sense.
  KALDI_ASSERT(posterior_scale > 0.0 && posterior_scale <= 1.0);
  KALDI_ASSERT(max_remembered_frames >= 0);
}

// The class constructed in this way should never be used.
OnlineIvectorExtractionInfo::OnlineIvectorExtractionInfo():
    ivector_period(0), num_gselect(0), min_post(0.0), posterior_scale(0.0),
    use_most_recent_ivector(false), max_remembered_frames(0) { }

OnlineIvectorExtractorAdaptationState::OnlineIvectorExtractorAdaptationState(
    const OnlineIvectorExtractorAdaptationState &other):
    cmvn_state(other.cmvn_state), ivector_stats(other.ivector_stats) { }


void OnlineIvectorExtractorAdaptationState::LimitFrames(
    BaseFloat max_remembered_frames, BaseFloat posterior_scale) {
  KALDI_ASSERT(max_remembered_frames >= 0);
  KALDI_ASSERT(cmvn_state.frozen_state.NumRows() == 0);
  if (cmvn_state.speaker_cmvn_stats.NumRows() != 0) {
    int32 feat_dim = cmvn_state.speaker_cmvn_stats.NumCols() - 1;
    BaseFloat count = cmvn_state.speaker_cmvn_stats(0, feat_dim);
    if (count > max_remembered_frames)
      cmvn_state.speaker_cmvn_stats.Scale(max_remembered_frames / count);
  }
  // the stats for the iVector have been scaled by info_.posterior_scale,
  // so we need to take this in account when setting the target count.
  BaseFloat max_remembered_frames_scaled =
      max_remembered_frames * posterior_scale;
  if (ivector_stats.Count() > max_remembered_frames_scaled) {
    ivector_stats.Scale(max_remembered_frames_scaled /
                        ivector_stats.Count());
  }  
}

int32 OnlineIvectorFeature::Dim() const {
  return info_.extractor.IvectorDim();
}

bool OnlineIvectorFeature::IsLastFrame(int32 frame) const {
  // Note: it might be more logical to return, say, lda_->IsLastFrame()
  // since this is the feature the iVector extractor directly consumes,
  // but it will give the same answer as base_->IsLastFrame() anyway.
  // [note: the splicing component pads at begin and end so it always
  // returns the same number of frames as its input.]
  return base_->IsLastFrame(frame);
}

int32 OnlineIvectorFeature::NumFramesReady() const {
  KALDI_ASSERT(lda_ != NULL);
  return lda_->NumFramesReady();
}

void OnlineIvectorFeature::UpdateStatsUntilFrame(int32 frame) {
  KALDI_ASSERT(frame >= 0 && frame < this->NumFramesReady());

  int32 feat_dim = lda_normalized_->Dim(),
      ivector_period = info_.ivector_period;
  int32 needed_frame = ivector_period * (frame / ivector_period);
  // needed_frame is the frame on which we'll estimate the iVector that
  // we need to output, so we'll need this regardless of
  // whether info_.use_most_recent_ivector == true.  This will end up
  // in cur_ivector_.
  
  int32 num_cg_iters = 15;  // I don't believe this is very important, so it's
                            // not configurable from the command line for now.
  
  Vector<BaseFloat> feat(feat_dim),  // features given to iVector extractor
      log_likes(info_.diag_ubm.NumGauss());

  for (; num_frames_stats_ <= frame; num_frames_stats_++) {
    int32 t = num_frames_stats_;  // Frame whose stats we want to get.
    lda_normalized_->GetFrame(t, &feat);
    info_.diag_ubm.LogLikelihoods(feat, &log_likes);
    // "posterior" stores the pruned posteriors for Gaussians in the UBM.
    std::vector<std::pair<int32, BaseFloat> > posterior;
    tot_ubm_loglike_ += VectorToPosteriorEntry(log_likes, info_.num_gselect,
                                               info_.min_post, &posterior);
    for (size_t i = 0; i < posterior.size(); i++)
      posterior[i].second *= info_.posterior_scale;
    lda_->GetFrame(t, &feat); // get feature without CMN.
    ivector_stats_.AccStats(info_.extractor, feat, posterior);
    
    if ((!info_.use_most_recent_ivector && t % ivector_period == 0) ||
        (info_.use_most_recent_ivector && t == needed_frame)) {
      ivector_stats_.GetIvector(num_cg_iters, &current_ivector_);
      if (!info_.use_most_recent_ivector) {  // need to cache iVectors.
        int32 ivec_index = t / ivector_period;
        KALDI_ASSERT(ivec_index == static_cast<int32>(ivectors_history_.size()));
        ivectors_history_.push_back(new Vector<BaseFloat>(current_ivector_));
      }
    }
  }
}

void OnlineIvectorFeature::GetFrame(int32 frame,
                                    VectorBase<BaseFloat> *feat) {
  UpdateStatsUntilFrame(frame);
  KALDI_ASSERT(feat->Dim() == this->Dim());

  if (info_.use_most_recent_ivector) {
    // use the most recent iVector we have, even if 'frame' is significantly in
    // the past.
    feat->CopyFromVec(current_ivector_);
    // Subtract the prior-mean from the first dimension of the output feature so
    // it's approximately zero-mean.
    (*feat)(0) -= info_.extractor.PriorOffset();
  } else {
    int32 i = frame / info_.ivector_period;  // rounds down.
    // if the following fails, UpdateStatsUntilFrame would have a bug.
    KALDI_ASSERT(static_cast<size_t>(i) <  ivectors_history_.size());
    feat->CopyFromVec(*(ivectors_history_[i]));
    (*feat)(0) -= info_.extractor.PriorOffset();
  }
}

void OnlineIvectorFeature::PrintDiagnostics() const {
  if (num_frames_stats_ == 0) {
    KALDI_VLOG(3) << "Processed no data.";
  } else {
    KALDI_VLOG(3) << "UBM log-likelihood was "
                  << (tot_ubm_loglike_ / num_frames_stats_)
                  << " per frame, over " << num_frames_stats_
                  << " frames.";
    KALDI_VLOG(3) << "By the end of the utterance, objf change/frame "
                  << "from estimating iVector (vs. default) was "
                  << ivector_stats_.ObjfChange(current_ivector_);
  }
}

OnlineIvectorFeature::~OnlineIvectorFeature() {
  PrintDiagnostics();
  // Delete objects owned here.
  delete lda_normalized_;
  delete splice_normalized_;
  delete cmvn_;
  delete lda_;
  delete splice_;
  // base_ is not owned here so don't delete it.
  for (size_t i = 0; i < ivectors_history_.size(); i++)
    delete ivectors_history_[i];
}

void OnlineIvectorFeature::GetAdaptationState(
    OnlineIvectorExtractorAdaptationState *adaptation_state) const {
  // Note: the following call will work even if cmvn_->NumFramesReady() == 0; in
  // that case it will return the unmodified adaptation state that cmvn_ was
  // initialized with.
  cmvn_->GetState(cmvn_->NumFramesReady() - 1,
                  &(adaptation_state->cmvn_state));
  adaptation_state->ivector_stats = ivector_stats_;
  adaptation_state->LimitFrames(info_.max_remembered_frames,
                                info_.posterior_scale);
}


OnlineIvectorFeature::OnlineIvectorFeature(
    const OnlineIvectorExtractionInfo &info,
    OnlineFeatureInterface *base_feature):
    info_(info), base_(base_feature),
    ivector_stats_(info_.extractor.IvectorDim(), info_.extractor.PriorOffset()),
    num_frames_stats_(0) {
  info.Check();
  KALDI_ASSERT(base_feature != NULL);
  splice_ = new OnlineSpliceFrames(info_.splice_opts, base_);
  lda_ = new OnlineTransform(info.lda_mat, splice_);
  OnlineCmvnState naive_cmvn_state(info.global_cmvn_stats);
  // Note: when you call this constructor the CMVN state knows nothing
  // about the speaker.  If you want to inform this class about more specific
  // adaptation state, call this->SetAdaptationState(), most likely derived
  // from a call to GetAdaptationState() from a previous object of this type.
  cmvn_ = new OnlineCmvn(info.cmvn_opts, naive_cmvn_state, base_);
  splice_normalized_ = new OnlineSpliceFrames(info_.splice_opts, cmvn_);
  lda_normalized_ = new OnlineTransform(info.lda_mat, splice_normalized_);

  // Set the iVector to its default value, [ prior_offset, 0, 0, ... ].
  current_ivector_.Resize(info_.extractor.IvectorDim());
  current_ivector_(0) = info_.extractor.PriorOffset(); 
}

void OnlineIvectorFeature::SetAdaptationState(
    const OnlineIvectorExtractorAdaptationState &adaptation_state) {
  KALDI_ASSERT(num_frames_stats_ == 0 &&
               "SetAdaptationState called after frames were processed.");
  KALDI_ASSERT(ivector_stats_.IvectorDim() ==
               adaptation_state.ivector_stats.IvectorDim());
  ivector_stats_ = adaptation_state.ivector_stats;
  cmvn_->SetState(adaptation_state.cmvn_state);
}


}  // namespace kaldi
