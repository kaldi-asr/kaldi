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
  online_cmvn_iextractor = config.online_cmvn_iextractor;
  ivector_period = config.ivector_period;
  num_gselect = config.num_gselect;
  min_post = config.min_post;
  posterior_scale = config.posterior_scale;
  max_count = config.max_count;
  num_cg_iters = config.num_cg_iters;
  use_most_recent_ivector = config.use_most_recent_ivector;
  greedy_ivector_extractor = config.greedy_ivector_extractor;
  if (greedy_ivector_extractor && !use_most_recent_ivector) {
    KALDI_WARN << "--greedy-ivector-extractor=true implies "
               << "--use-most-recent-ivector=true";
    use_most_recent_ivector = true;
  }
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

int32 OnlineIvectorExtractionInfo::ExpectedFeatureDim() const {
  int32 num_splice = 1 + splice_opts.left_context + splice_opts.right_context,
      full_dim = lda_mat.NumCols();
  if (!(full_dim % num_splice == 0 || full_dim % num_splice == 1)){
    KALDI_WARN << "Error getting expected feature dimension: full-dim = "
               << full_dim << ", num-splice = " << num_splice;
  }
  return full_dim / num_splice;
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
    online_cmvn_iextractor(false), ivector_period(0), num_gselect(0), min_post(0.0), posterior_scale(0.0),
    use_most_recent_ivector(true), greedy_ivector_extractor(false),
    max_remembered_frames(0) { }

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

void OnlineIvectorExtractorAdaptationState::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<OnlineIvectorExtractorAdaptationState>");  // magic string.
  WriteToken(os, binary, "<CmvnState>");
  cmvn_state.Write(os, binary);
  WriteToken(os, binary, "<IvectorStats>");
  ivector_stats.Write(os, binary);
  WriteToken(os, binary, "</OnlineIvectorExtractorAdaptationState>");
}

void OnlineIvectorExtractorAdaptationState::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<OnlineIvectorExtractorAdaptationState>");  // magic string.
  ExpectToken(is, binary, "<CmvnState>");
  cmvn_state.Read(is, binary);
  ExpectToken(is, binary, "<IvectorStats>");
  ivector_stats.Read(is, binary);
  ExpectToken(is, binary, "</OnlineIvectorExtractorAdaptationState>");
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

BaseFloat OnlineIvectorFeature::FrameShiftInSeconds() const {
  return lda_->FrameShiftInSeconds();
}

void OnlineIvectorFeature::UpdateFrameWeights(
    const std::vector<std::pair<int32, BaseFloat> > &delta_weights) {
  // add the elements to delta_weights_, which is a priority queue.  The top
  // element of the priority queue is the lowest numbered frame (we ensured this
  // by making the comparison object std::greater instead of std::less).  Adding
  // elements from top (lower-numbered frames) to bottom (higher-numbered
  // frames) should be most efficient, assuming it's a heap internally.  So we
  // go forward not backward in delta_weights while adding.
  for (size_t i = 0; i < delta_weights.size(); i++) {
    delta_weights_.push(delta_weights[i]);
    int32 frame = delta_weights[i].first;
    KALDI_ASSERT(frame >= 0);
    if (frame > most_recent_frame_with_weight_)
      most_recent_frame_with_weight_ = frame;
  }
  delta_weights_provided_ = true;
}


BaseFloat OnlineIvectorFeature::GetMinPost(BaseFloat weight) const {
  BaseFloat min_post = info_.min_post;
  BaseFloat abs_weight = fabs(weight);
  // If we return 0.99, it will have the same effect as just picking the
  // most probable Gaussian on that frame.
  if (abs_weight == 0.0)
    return 0.99;   // I don't anticipate reaching here.
  min_post /= abs_weight;
  if (min_post > 0.99)
    min_post = 0.99;
  return min_post;
}

void OnlineIvectorFeature::UpdateStatsForFrames(
    const std::vector<std::pair<int32, BaseFloat> > &frame_weights_in) {

  std::vector<std::pair<int32, BaseFloat> > frame_weights(frame_weights_in);
  // Remove duplicates of frames.
  MergePairVectorSumming(&frame_weights);

  if (frame_weights.empty())
    return;

  int32 num_frames = static_cast<int32>(frame_weights.size());
  int32 feat_dim = lda_normalized_->Dim();
  Matrix<BaseFloat> feats(num_frames, feat_dim, kUndefined),
      log_likes;

  std::vector<int32> frames;
  frames.reserve(frame_weights.size());
  for (int32 i = 0; i < num_frames; i++)
    frames.push_back(frame_weights[i].first);
  lda_normalized_->GetFrames(frames, &feats);

  info_.diag_ubm.LogLikelihoods(feats, &log_likes);

  // "posteriors" stores, for each frame index in the range of frames, the
  // pruned posteriors for the Gaussians in the UBM.
  std::vector<std::vector<std::pair<int32, BaseFloat> > > posteriors(num_frames);
  for (int32 i = 0; i < num_frames; i++) {
    std::vector<std::pair<int32, BaseFloat> > &posterior = posteriors[i];
    BaseFloat weight = frame_weights[i].second;
    if (weight != 0.0) {
      tot_ubm_loglike_ += weight *
          VectorToPosteriorEntry(log_likes.Row(i), info_.num_gselect,
                                 GetMinPost(weight), &posterior);
      for (size_t j = 0; j < posterior.size(); j++)
        posterior[j].second *= info_.posterior_scale * weight;
    }
  }

  if (! info_.online_cmvn_iextractor) {
    lda_->GetFrames(frames, &feats);  // default, get features without OnlineCmvn
  } else {
    lda_normalized_->GetFrames(frames, &feats); // get features with OnlineCmvn
  }
  ivector_stats_.AccStats(info_.extractor, feats, posteriors);
}


void OnlineIvectorFeature::UpdateStatsUntilFrame(int32 frame) {
  KALDI_ASSERT(frame >= 0 && frame < this->NumFramesReady() &&
               !delta_weights_provided_);
  updated_with_no_delta_weights_ = true;

  int32 ivector_period = info_.ivector_period;
  int32 num_cg_iters = info_.num_cg_iters;

  std::vector<std::pair<int32, BaseFloat> > frame_weights;

  for (; num_frames_stats_ <= frame; num_frames_stats_++) {
    int32 t = num_frames_stats_;
    BaseFloat frame_weight = 1.0;
    frame_weights.push_back(std::pair<int32, BaseFloat>(t, frame_weight));
    if ((!info_.use_most_recent_ivector && t % ivector_period == 0) ||
        (info_.use_most_recent_ivector && t == frame)) {
      // The call below to UpdateStatsForFrames() is equivalent to doing, for
      // all valid indexes i:
      //  UpdateStatsForFrame(cur_start_frame + i, frame_weights[i])
      UpdateStatsForFrames(frame_weights);
      frame_weights.clear();
      ivector_stats_.GetIvector(num_cg_iters, &current_ivector_);
      if (!info_.use_most_recent_ivector) {  // need to cache iVectors.
        int32 ivec_index = t / ivector_period;
        KALDI_ASSERT(ivec_index == static_cast<int32>(ivectors_history_.size()));
        ivectors_history_.push_back(new Vector<BaseFloat>(current_ivector_));
      }
    }
  }
  if (!frame_weights.empty())
    UpdateStatsForFrames(frame_weights);
}

void OnlineIvectorFeature::UpdateStatsUntilFrameWeighted(int32 frame) {
  KALDI_ASSERT(frame >= 0 && frame < this->NumFramesReady() &&
               delta_weights_provided_ &&
               ! updated_with_no_delta_weights_ &&
               frame <= most_recent_frame_with_weight_);
  bool debug_weights = false;

  int32 ivector_period = info_.ivector_period;
  int32 num_cg_iters = info_.num_cg_iters;

  std::vector<std::pair<int32, BaseFloat> > frame_weights;
  frame_weights.reserve(delta_weights_.size());

  for (; num_frames_stats_ <= frame; num_frames_stats_++) {
    int32 t = num_frames_stats_;
    // Instead of just updating frame t, we update all frames that need updating
    // with index <= t, in case old frames were reclassified as silence/nonsilence.
    while (!delta_weights_.empty() &&
           delta_weights_.top().first <= t) {
      int32 frame = delta_weights_.top().first;
      BaseFloat weight = delta_weights_.top().second;
      frame_weights.push_back(delta_weights_.top());
      delta_weights_.pop();
      if (debug_weights) {
        if (current_frame_weight_debug_.size() <= frame)
          current_frame_weight_debug_.resize(frame + 1, 0.0);
        current_frame_weight_debug_[frame] += weight;
      }
    }
    if ((!info_.use_most_recent_ivector && t % ivector_period == 0) ||
        (info_.use_most_recent_ivector && t == frame)) {
      UpdateStatsForFrames(frame_weights);
      frame_weights.clear();
      ivector_stats_.GetIvector(num_cg_iters, &current_ivector_);
      if (!info_.use_most_recent_ivector) {  // need to cache iVectors.
        int32 ivec_index = t / ivector_period;
        KALDI_ASSERT(ivec_index == static_cast<int32>(ivectors_history_.size()));
        ivectors_history_.push_back(new Vector<BaseFloat>(current_ivector_));
      }
    }
  }
  if (!frame_weights.empty())
    UpdateStatsForFrames(frame_weights);
}


void OnlineIvectorFeature::GetFrame(int32 frame,
                                    VectorBase<BaseFloat> *feat) {
  int32 frame_to_update_until = (info_.greedy_ivector_extractor ?
                                 lda_->NumFramesReady() - 1 : frame);
  if (!delta_weights_provided_)  // No silence weighting.
    UpdateStatsUntilFrame(frame_to_update_until);
  else
    UpdateStatsUntilFrameWeighted(frame_to_update_until);

  KALDI_ASSERT(feat->Dim() == this->Dim());

  if (info_.use_most_recent_ivector) {
    KALDI_VLOG(5) << "due to --use-most-recent-ivector=true, using iVector "
                  << "from frame " << num_frames_stats_ << " for frame "
                  << frame;
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
                  << (tot_ubm_loglike_ / NumFrames())
                  << " per frame, over " << NumFrames()
                  << " frames.";

    Vector<BaseFloat> temp_ivector(current_ivector_);
    temp_ivector(0) -= info_.extractor.PriorOffset();

    KALDI_VLOG(2) << "By the end of the utterance, objf change/frame "
                  << "from estimating iVector (vs. default) was "
                  << ivector_stats_.ObjfChange(current_ivector_)
                  << " and iVector length was "
                  << temp_ivector.Norm(2.0);
  }
}

OnlineIvectorFeature::~OnlineIvectorFeature() {
  PrintDiagnostics();
  // Delete objects owned here.
  for (size_t i = 0; i < to_delete_.size(); i++)
    delete to_delete_[i];
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
    info_(info),
    base_(base_feature),
    ivector_stats_(info_.extractor.IvectorDim(),
                   info_.extractor.PriorOffset(),
                   info_.max_count),
    num_frames_stats_(0), delta_weights_provided_(false),
    updated_with_no_delta_weights_(false),
    most_recent_frame_with_weight_(-1), tot_ubm_loglike_(0.0) {
  info.Check();
  KALDI_ASSERT(base_feature != NULL);
  OnlineFeatureInterface *splice_feature = new OnlineSpliceFrames(info_.splice_opts, base_feature);
  to_delete_.push_back(splice_feature);
  OnlineFeatureInterface *lda_feature = new OnlineTransform(info.lda_mat, splice_feature);
  to_delete_.push_back(lda_feature);
  OnlineFeatureInterface *lda_cache_feature = new OnlineCacheFeature(lda_feature);
  lda_ = lda_cache_feature;
  to_delete_.push_back(lda_cache_feature);


  OnlineCmvnState naive_cmvn_state(info.global_cmvn_stats);
  // Note: when you call this constructor the CMVN state knows nothing
  // about the speaker.  If you want to inform this class about more specific
  // adaptation state, call this->SetAdaptationState(), most likely derived
  // from a call to GetAdaptationState() from a previous object of this type.
  cmvn_ = new OnlineCmvn(info.cmvn_opts, naive_cmvn_state, base_feature);
  to_delete_.push_back(cmvn_);

  OnlineFeatureInterface *splice_normalized =
      new OnlineSpliceFrames(info_.splice_opts, cmvn_),
      *lda_normalized =
      new OnlineTransform(info.lda_mat, splice_normalized),
      *cache_normalized = new OnlineCacheFeature(lda_normalized);
  lda_normalized_ = cache_normalized;

  to_delete_.push_back(splice_normalized);
  to_delete_.push_back(lda_normalized);
  to_delete_.push_back(cache_normalized);

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

BaseFloat OnlineIvectorFeature::UbmLogLikePerFrame() const {
  if (NumFrames() == 0) return 0;
  else return tot_ubm_loglike_ / NumFrames();
}

BaseFloat OnlineIvectorFeature::ObjfImprPerFrame() const {
  return ivector_stats_.ObjfChange(current_ivector_);
}


OnlineSilenceWeighting::OnlineSilenceWeighting(
    const TransitionModel &trans_model,
    const OnlineSilenceWeightingConfig &config,
    int32 frame_subsampling_factor):
    trans_model_(trans_model), config_(config),
    frame_subsampling_factor_(frame_subsampling_factor),
    num_frames_output_and_correct_(0) {
  KALDI_ASSERT(frame_subsampling_factor_ >= 1);
  std::vector<int32> silence_phones;
  SplitStringToIntegers(config.silence_phones_str, ":,", false,
                        &silence_phones);
  for (size_t i = 0; i < silence_phones.size(); i++)
    silence_phones_.insert(silence_phones[i]);
}


template <typename FST>
void OnlineSilenceWeighting::ComputeCurrentTraceback(
    const LatticeFasterOnlineDecoderTpl<FST> &decoder,
    bool use_final_probs) {
  int32 num_frames_decoded = decoder.NumFramesDecoded(),
      num_frames_prev = frame_info_.size();
  // note, num_frames_prev is not the number of frames previously decoded,
  // it's the generally-larger number of frames that we were requested to
  // provide weights for.
  if (num_frames_prev < num_frames_decoded)
    frame_info_.resize(num_frames_decoded);
  if (num_frames_prev > num_frames_decoded &&
      frame_info_[num_frames_decoded].transition_id != -1)
    KALDI_ERR << "Number of frames decoded decreased";  // Likely bug

  if (num_frames_decoded == 0)
    return;
  int32 frame = num_frames_decoded - 1;
  typename LatticeFasterOnlineDecoderTpl<FST>::BestPathIterator iter =
      decoder.BestPathEnd(use_final_probs, NULL);
  while (frame >= 0) {
    LatticeArc arc;
    arc.ilabel = 0;
    while (arc.ilabel == 0)  // the while loop skips over input-epsilons
      iter = decoder.TraceBackBestPath(iter, &arc);
    // note, the iter.frame values are slightly unintuitively defined,
    // they are one less than you might expect.
    KALDI_ASSERT(iter.frame == frame - 1);

    if (frame_info_[frame].token == iter.tok) {
      // we know that the traceback from this point back will be identical, so
      // no point tracing back further.  Note: we are comparing memory addresses
      // of tokens of the decoder; this guarantees it's the same exact token
      // because tokens, once allocated on a frame, are only deleted, never
      // reallocated for that frame.
      break;
    }

    if (num_frames_output_and_correct_ > frame)
      num_frames_output_and_correct_ = frame;

    frame_info_[frame].token = iter.tok;
    frame_info_[frame].transition_id = arc.ilabel;
    frame--;
    // leave frame_info_.current_weight at zero for now (as set in the
    // constructor), reflecting that we haven't already output a weight for that
    // frame.
  }
}

template <typename FST>
void OnlineSilenceWeighting::ComputeCurrentTraceback(
    const LatticeIncrementalOnlineDecoderTpl<FST> &decoder,
    bool use_final_probs) {
  int32 num_frames_decoded = decoder.NumFramesDecoded(),
      num_frames_prev = frame_info_.size();
  // note, num_frames_prev is not the number of frames previously decoded,
  // it's the generally-larger number of frames that we were requested to
  // provide weights for.
  if (num_frames_prev < num_frames_decoded)
    frame_info_.resize(num_frames_decoded);
  if (num_frames_prev > num_frames_decoded &&
      frame_info_[num_frames_decoded].transition_id != -1)
    KALDI_ERR << "Number of frames decoded decreased";  // Likely bug

  if (num_frames_decoded == 0)
    return;
  int32 frame = num_frames_decoded - 1;
  typename LatticeIncrementalOnlineDecoderTpl<FST>::BestPathIterator iter =
      decoder.BestPathEnd(use_final_probs, NULL);
  while (frame >= 0) {
    LatticeArc arc;
    arc.ilabel = 0;
    while (arc.ilabel == 0)  // the while loop skips over input-epsilons
      iter = decoder.TraceBackBestPath(iter, &arc);
    // note, the iter.frame values are slightly unintuitively defined,
    // they are one less than you might expect.
    KALDI_ASSERT(iter.frame == frame - 1);

    if (frame_info_[frame].token == iter.tok) {
      // we know that the traceback from this point back will be identical, so
      // no point tracing back further.  Note: we are comparing memory addresses
      // of tokens of the decoder; this guarantees it's the same exact token,
      // because tokens, once allocated on a frame, are only deleted, never
      // reallocated for that frame.
      break;
    }

    if (num_frames_output_and_correct_ > frame)
      num_frames_output_and_correct_ = frame;

    frame_info_[frame].token = iter.tok;
    frame_info_[frame].transition_id = arc.ilabel;
    frame--;
    // leave frame_info_.current_weight at zero for now (as set in the
    // constructor), reflecting that we haven't already output a weight for that
    // frame.
  }
}


// Instantiate the template OnlineSilenceWeighting::ComputeCurrentTraceback().
template
void OnlineSilenceWeighting::ComputeCurrentTraceback<fst::Fst<fst::StdArc> >(
    const LatticeFasterOnlineDecoderTpl<fst::Fst<fst::StdArc> > &decoder,
    bool use_final_probs);
template
void OnlineSilenceWeighting::ComputeCurrentTraceback<fst::ConstGrammarFst >(
    const LatticeFasterOnlineDecoderTpl<fst::ConstGrammarFst > &decoder,
    bool use_final_probs);
template
void OnlineSilenceWeighting::ComputeCurrentTraceback<fst::VectorGrammarFst >(
    const LatticeFasterOnlineDecoderTpl<fst::VectorGrammarFst > &decoder,
    bool use_final_probs);

template
void OnlineSilenceWeighting::ComputeCurrentTraceback<fst::Fst<fst::StdArc> >(
    const LatticeIncrementalOnlineDecoderTpl<fst::Fst<fst::StdArc> > &decoder,
    bool use_final_probs);
template
void OnlineSilenceWeighting::ComputeCurrentTraceback<fst::ConstGrammarFst >(
    const LatticeIncrementalOnlineDecoderTpl<fst::ConstGrammarFst > &decoder,
    bool use_final_probs);
template
void OnlineSilenceWeighting::ComputeCurrentTraceback<fst::VectorGrammarFst >(
    const LatticeIncrementalOnlineDecoderTpl<fst::VectorGrammarFst > &decoder,
    bool use_final_probs);


void OnlineSilenceWeighting::GetDeltaWeights(
    int32 num_frames_ready, int32 first_decoder_frame,
    std::vector<std::pair<int32, BaseFloat> > *delta_weights) {
  // num_frames_ready is at the feature frame-rate, most of the code
  // in this function is at the decoder frame-rate.
  // round up, so we are sure to get weights for at least the frame
  // 'num_frames_ready - 1', and maybe one or two frames afterward.
  KALDI_ASSERT(num_frames_ready > first_decoder_frame || num_frames_ready == 0);
  int32 fs = frame_subsampling_factor_,
  num_decoder_frames_ready = (num_frames_ready - first_decoder_frame + fs - 1) / fs;

  const int32 max_state_duration = config_.max_state_duration;
  const BaseFloat silence_weight = config_.silence_weight;

  delta_weights->clear();

  int32 prev_num_frames_processed = frame_info_.size();
  if (frame_info_.size() < static_cast<size_t>(num_decoder_frames_ready))
    frame_info_.resize(num_decoder_frames_ready);

  // Don't go further backward into the past then 100 frames before the most
  // recent frame previously than 100 frames when modifying the traceback.
  // C.f. the value 200 in template
  // OnlineGenericBaseFeature<C>::OnlineGenericBaseFeature in online-feature.cc,
  // which needs to be more than this value of 100 plus the amount of context
  // that LDA might use plus the chunk size we're likely to decode in one time.
  // The user can always increase the value of --max-feature-vectors in case one
  // of these conditions is broken.  Search for ONLINE_IVECTOR_LIMIT in
  // online-feature.cc
  int32 begin_frame = std::max<int32>(0, prev_num_frames_processed - 100),
      frames_out = static_cast<int32>(frame_info_.size()) - begin_frame;
  // frames_out is the number of frames we will output.
  KALDI_ASSERT(frames_out >= 0);
  std::vector<BaseFloat> frame_weight(frames_out, 1.0);
  // we will set frame_weight to the value silence_weight for silence frames and
  // for transition-ids that repeat with duration > max_state_duration.  Frames
  // newer than the most recent traceback will get a weight equal to the weight
  // for the most recent frame in the traceback; or the silence weight, if there
  // is no traceback at all available yet.

  // First treat some special cases.
  if (frames_out == 0)  // Nothing to output.
    return;
  if (frame_info_[begin_frame].transition_id == -1) {
    // We do not have any traceback at all within the frames we are to output...
    // find the most recent weight that we output and apply the same weight to
    // all the new output; or output the silence weight, if nothing was output.
    BaseFloat weight = (begin_frame == 0 ? silence_weight :
                        frame_info_[begin_frame - 1].current_weight);
    for (int32 offset = 0; offset < frames_out; offset++)
      frame_weight[offset] = weight;
  } else {
    int32 current_run_start_offset = 0;
    for (int32 offset = 0; offset < frames_out; offset++) {
      int32 frame = begin_frame + offset;
      int32 transition_id = frame_info_[frame].transition_id;
      if (transition_id == -1) {
        // this frame does not yet have a decoder traceback, so just
        // duplicate the silence/non-silence status of the most recent
        // frame we have a traceback for (probably a reasonable guess).
        frame_weight[offset] = frame_weight[offset - 1];
      } else {
        int32 phone = trans_model_.TransitionIdToPhone(transition_id);
        bool is_silence = (silence_phones_.count(phone) != 0);
        if (is_silence)
          frame_weight[offset] = silence_weight;
        // now deal with max-duration issues.
        if (max_state_duration > 0 &&
            (offset + 1 == frames_out ||
             transition_id != frame_info_[frame + 1].transition_id)) {
          // If this is the last frame of a run...
          int32 run_length = offset - current_run_start_offset + 1;
          if (run_length >= max_state_duration) {
            // treat runs of the same transition-id longer than the max, as
            // silence, even if they were not silence.
            for (int32 offset2 = current_run_start_offset;
                 offset2 <= offset; offset2++)
              frame_weight[offset2] = silence_weight;
          }
          if (offset + 1 < frames_out)
            current_run_start_offset = offset + 1;
        }
      }
    }
  }
  // Now commit the stats...
  for (int32 offset = 0; offset < frames_out; offset++) {
    int32 frame = begin_frame + offset;
    BaseFloat old_weight = frame_info_[frame].current_weight,
        new_weight = frame_weight[offset],
        weight_diff = new_weight - old_weight;
    frame_info_[frame].current_weight = new_weight;
    // Even if the delta-weight is zero for the last frame, we provide it,
    // because the identity of the most recent frame with a weight is used in
    // some debugging/checking code.
    if (weight_diff != 0.0 || offset + 1 == frames_out) {
      KALDI_VLOG(6) << "Weight for frame " << frame << " changing from "
                    << old_weight << " to " << new_weight;
      for(int32 i = 0; i < frame_subsampling_factor_; i++) {
        int32 input_frame = first_decoder_frame + (frame * frame_subsampling_factor_) + i;
        delta_weights->push_back(std::make_pair(input_frame, weight_diff));
      }
    }
  }
}

void OnlineSilenceWeighting::GetNonsilenceFrames(
    int32 num_frames_ready, int32 first_decoder_frame,
    std::vector<int32> *frames) {
  // num_frames_ready is at the feature frame-rate, most of the code
  // in this function is at the decoder frame-rate.
  // round up, so we are sure to get weights for at least the frame
  // 'num_frames_ready - 1', and maybe one or two frames afterward.
  KALDI_ASSERT(num_frames_ready > first_decoder_frame || num_frames_ready == 0);
  int32 fs = frame_subsampling_factor_,
  num_decoder_frames_ready = (num_frames_ready - first_decoder_frame + fs - 1) / fs;

  frames->clear();

  int32 prev_num_frames_processed = frame_info_.size();
  if (frame_info_.size() < static_cast<size_t>(num_decoder_frames_ready))
    frame_info_.resize(num_decoder_frames_ready);

  // Don't go further backward into the past then 500 frames before the most
  // recent frame
  int32 begin_frame = std::max<int32>(0, prev_num_frames_processed - 500),
      frames_out = static_cast<int32>(frame_info_.size()) - begin_frame;
  // frames_out is the number of frames we will output.
  KALDI_ASSERT(frames_out >= 0);

  for (int32 offset = 0; offset < frames_out; offset++) {
    int32 frame = begin_frame + offset;
    int32 transition_id = frame_info_[frame].transition_id;
    if (transition_id != -1) {
      int32 phone = trans_model_.TransitionIdToPhone(transition_id);
      bool is_silence = (silence_phones_.count(phone) != 0);
      if (!is_silence) {
        frames->push_back(frame);
      }
    }
  }
}


}  // namespace kaldi
