// online2/online-gmm-decoding.cc

// Copyright    2013-2014  Johns Hopkins University (author: Daniel Povey)

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

#include "online2/online-gmm-decoding.h"
#include "lat/lattice-functions.h"
#include "lat/determinize-lattice-pruned.h"

namespace kaldi {


SingleUtteranceGmmDecoder::SingleUtteranceGmmDecoder(
    const OnlineGmmDecodingConfig &config,
    const OnlineGmmDecodingModels &models,                            
    const OnlineFeaturePipeline &feature_prototype,
    const fst::Fst<fst::StdArc> &fst,
    const SpeakerAdaptationState &adaptation_state):
    config_(config), models_(models),
    feature_pipeline_(feature_prototype.New()),
    orig_adaptation_state_(adaptation_state),
    adaptation_state_(adaptation_state),
    decoder_(fst, config.faster_decoder_opts) {
  if (!SplitStringToIntegers(config_.silence_phones, ":", false,
                             &silence_phones_))
    KALDI_ERR << "Bad --silence-phones option '"
              << config_.silence_phones << "'";
  SortAndUniq(&silence_phones_);
  feature_pipeline_->SetTransform(adaptation_state_.transform);
  decoder_.InitDecoding();
}
    

/** Advance the first-pass decoding as far as we can. */
void SingleUtteranceGmmDecoder::AdvanceFirstPass() {
  
  const AmDiagGmm &am_gmm = (HaveTransform() ? models_.GetModel() :
                             models_.GetOnlineAlignmentModel());

  // The decodable object is lightweight, we lose nothing
  // from constructing it each time we want to decode more of the
  // input.
  DecodableDiagGmmScaledOnline decodable(am_gmm,
                                         models_.GetTransitionModel(),
                                         config_.acoustic_scale,
                                         feature_pipeline_);

  // This will decode as many frames as are currently available.
  decoder_.DecodeNonblocking(&decodable);  
}

// We exclude the silence phones from the Gaussian posteriors.
bool SingleUtteranceGmmDecoder::GetGaussianPosteriors(bool end_of_utterance,
                                                      GaussPost *gpost) {
  // Gets the Gaussian-level posteriors for this utterance, using whatever
  // features and model we are currently decoding with.  We'll use these
  // to estimate basis-fMLLR with.
  if (decoder_.NumFramesDecoded() == 0) {
    KALDI_WARN << "You have decoded no data so cannot estimate fMLLR.";
    return false;
  }

  // Note: we'll just use whatever acoustic scaling factor we were decoding
  // with.  This is in the lattice that we get from decoder_.GetRawLattice().
  Lattice raw_lat;
  decoder_.GetRawLattice(&raw_lat, end_of_utterance);

  // At this point we could rescore the lattice if we wanted, and
  // this might improve the accuracy on long utterances that were
  // the first utterance of that speaker, if we had already
  // estimated the fMLLR by the time we reach this code (e.g. this
  // was the second call).  We don't do this right now.
  
  KALDI_ASSERT(config_.fmllr_lattice_beam > 0.0);
  PruneLattice(config_.fmllr_lattice_beam, &raw_lat);
  Lattice det_lat; // lattice-determinized lattice-- represent this as Lattice
                   // not CompactLattice, as LatticeForwardBackward() does not
                   // accept CompactLattice.


  // TODO: after Guoguo finishes wrapper, convert to phone version.
  // Ignore the return status of DeterminizeLatticePruned; it doesn't matter
  // much if determinization finishes sooner than the beam.
  fst::DeterminizeLatticePruned(raw_lat,
                                double(config_.fmllr_lattice_beam),
                                &det_lat);
  
  if (det_lat.NumStates() == 0) {
    // Do nothing if the lattice is empty.  This should not happen.
    KALDI_WARN << "Got empty lattice.  Not estimating fMLLR.";
    return false;
  }
  
  TopSortLatticeIfNeeded(&det_lat);
  
  // Note: the acoustic scale we use here is whatever we decoded with.
  Posterior post;
  BaseFloat tot_fb_like = LatticeForwardBackward(det_lat, &post);

  KALDI_VLOG(3) << "Lattice forward-backward likelihood was "
                << (tot_fb_like / post.size()) << " per frame over " << post.size()
                << " frames.";

  ConstIntegerSet<int32> silence_set(silence_phones_);  // faster lookup
  const TransitionModel &trans_model = models_.GetTransitionModel();
  WeightSilencePost(trans_model, silence_set,
                    config_.silence_weight, &post);  
  
  const AmDiagGmm &am_gmm = (HaveTransform() ? models_.GetModel() :
                             models_.GetOnlineAlignmentModel());


  Posterior pdf_post;
  ConvertPosteriorToPdfs(trans_model, post, &pdf_post);
  
  Vector<BaseFloat> feat(feature_pipeline_->Dim());

  double tot_like = 0.0, tot_weight = 0.0;
  gpost->resize(pdf_post.size());
  for (size_t i = 0; i < pdf_post.size(); i++) {
    feature_pipeline_->GetFrame(i, &feat);
    for (size_t j = 0; j < pdf_post[i].size(); j++) {
      int32 pdf_id = pdf_post[i][j].first;
      BaseFloat weight = pdf_post[i][j].second;
      const DiagGmm &gmm = am_gmm.GetPdf(pdf_id);
      Vector<BaseFloat> this_post_vec;
      BaseFloat like = gmm.ComponentPosteriors(feat, &this_post_vec);
      this_post_vec.Scale(weight);
      tot_like += like * weight;
      tot_weight += weight;
      (*gpost)[i].push_back(std::make_pair(pdf_id, this_post_vec));
    }
  }
  KALDI_VLOG(3) << "Average likelihood weighted by posterior was "
                << (tot_like / tot_weight) << " over" << tot_weight
                << " frames (after downweighting silence).";  
  return true;
}


void SingleUtteranceGmmDecoder::EstimateFmllr(bool end_of_utterance) {
  if (decoder_.NumFramesDecoded() == 0) {
    KALDI_WARN << "You have decoded no data so cannot estimate fMLLR.";
  }


  GaussPost gpost;
  GetGaussianPosteriors(end_of_utterance, &gpost);
  
  FmllrDiagGmmAccs &spk_stats = adaptation_state_.spk_stats;
  
  int32 dim = feature_pipeline_->Dim();
  if (spk_stats.beta_ !=
      orig_adaptation_state_.spk_stats.beta_) {
    // This could happen if the user called EstimateFmllr() twice on the
    // same utterance... we don't want to count any stats twice so we
    // have to reset the stats to what they were before this utterance
    // (possibly empty).
    spk_stats = orig_adaptation_state_.spk_stats;
  }
  if (spk_stats.Dim() == 0)
    spk_stats.Init(dim);
  
  Matrix<BaseFloat> empty_transform;
  feature_pipeline_->SetTransform(empty_transform);
  Vector<BaseFloat> feat(dim);

  if (adaptation_state_.transform.NumRows() == 0) {
    // If this is the first time we're estimating fMLLR, freeze the CMVN to its
    // current value.  It doesn't matter too much what value this is, since we
    // have already computed the Gaussian-level alignments (it may have a small
    // effect if the basis is very small and doesn't include an offset as part
    // of the transform).
    feature_pipeline_->FreezeCmvn();
  }
  
  // GetModel() returns the model to be used for estimating
  // transforms.
  const AmDiagGmm &am_gmm = models_.GetModel();
  
  for (size_t i = 0; i < gpost.size(); i++) {
    feature_pipeline_->GetFrame(i, &feat);    
    for (size_t j = 0; j < gpost[i].size(); j++) {
      int32 pdf_id = gpost[i][j].first; // caution: this gpost has pdf-id
                                        // instead of transition-id, which is
                                        // unusual.
      const Vector<BaseFloat> &posterior(gpost[i][j].second);
      spk_stats.AccumulateFromPosteriors(am_gmm.GetPdf(pdf_id),
                                         feat, posterior);
    }
  }
  
  const BasisFmllrEstimate &basis = models_.GetFmllrBasis();
  if (basis.Dim() == 0)
    KALDI_ERR << "In order to estimate fMLLR, you need to supply the "
              << "--fmllr-basis option.";
  Vector<BaseFloat> basis_coeffs;
  BaseFloat impr = basis.ComputeTransform(spk_stats,
                                          &adaptation_state_.transform,
                                          &basis_coeffs, config_.basis_opts);
  KALDI_VLOG(3) << "Objective function improvement from basis-fMLLR is "
                << (impr / spk_stats.beta_) << " per frame, over "
                << spk_stats.beta_ << " frames, #params estimated is "
                << basis_coeffs.Dim();
  feature_pipeline_->SetTransform(adaptation_state_.transform);
}


bool SingleUtteranceGmmDecoder::HaveTransform() const {
  return (feature_pipeline_->HaveFmllrTransform());
}

void SingleUtteranceGmmDecoder::GetAdaptationState(
    SpeakerAdaptationState *adaptation_state) const {
  *adaptation_state = adaptation_state_;
  feature_pipeline_->GetCmvnState(&adaptation_state->cmvn_state);
}

bool SingleUtteranceGmmDecoder::RescoringIsNeeded() const {
  if (orig_adaptation_state_.transform.NumRows() !=
      adaptation_state_.transform.NumRows()) return true;  // fMLLR was estimated
  if (!orig_adaptation_state_.transform.ApproxEqual(
          adaptation_state_.transform)) return true;  // fMLLR was re-estimated
  if (adaptation_state_.transform.NumRows() != 0 &&
      &models_.GetModel() != &models_.GetFinalModel())
    return true; // we have an fMLLR transform, and a discriminatively estimated
                 // model which differs from the one used to estimate fMLLR.
  return false;
}

SingleUtteranceGmmDecoder::~SingleUtteranceGmmDecoder() {
  delete feature_pipeline_;
}

bool SingleUtteranceGmmDecoder::EndpointDetected(
    const OnlineEndpointConfig &config) const {
  Lattice best_path;
  const bool end_of_utterance = false;
  if (decoder_.NumFramesDecoded() == 0)
    return false;
  decoder_.GetBestPath(&best_path, end_of_utterance);
  const TransitionModel &tmodel = models_.GetTransitionModel();
  return kaldi::EndpointDetected(tmodel, best_path, config,
                                 feature_pipeline_->FrameShiftInSeconds(),
                                 decoder_.FinalRelativeCost());
}

void SingleUtteranceGmmDecoder::GetLattice(bool rescore_if_needed,
                                           bool end_of_utterance,
                                           CompactLattice *clat) const {
  Lattice lat;
  double lat_beam = config_.faster_decoder_opts.lattice_beam;
  decoder_.GetRawLattice(&lat, end_of_utterance);
  if (rescore_if_needed && RescoringIsNeeded()) {
    DecodableDiagGmmScaledOnline decodable(models_.GetFinalModel(),
                                           models_.GetTransitionModel(),
                                           config_.acoustic_scale,
                                           feature_pipeline_);

    if (!kaldi::RescoreLattice(&decodable, &lat))
      KALDI_WARN << "Error rescoring lattice";
  }
  PruneLattice(lat_beam, &lat);

  TopSortLatticeIfNeeded(&lat);
  Invert(&lat);
  // TODO: after Guoguo finishes wrapper, convert to phone version.
  fst::DeterminizeLatticePruned(lat, lat_beam, clat);
}

OnlineGmmDecodingModels::OnlineGmmDecodingModels(
    const OnlineGmmDecodingConfig &config) {
  KALDI_ASSERT(!config.model_rxfilename.empty() &&
               "You must supply the --model option");

  {
    bool binary;
    Input ki(config.model_rxfilename, &binary);
    tmodel_.Read(ki.Stream(), binary);
    model_.Read(ki.Stream(), binary);
  }
  
  if (!config.online_alimdl_rxfilename.empty()) {
    bool binary;
    Input ki(config.online_alimdl_rxfilename, &binary);
    TransitionModel tmodel;
    tmodel.Read(ki.Stream(), binary);
    if (!tmodel.Compatible(tmodel_))
      KALDI_ERR << "Incompatible models given to the --model and "
                << "--online-alignment-model options";
    online_alignment_model_.Read(ki.Stream(), binary);
  }

  if (!config.rescore_model_rxfilename.empty()) {
    bool binary;
    Input ki(config.rescore_model_rxfilename, &binary);
    TransitionModel tmodel;
    tmodel.Read(ki.Stream(), binary);
    if (!tmodel.Compatible(tmodel_))
      KALDI_ERR << "Incompatible models given to the --model and "
                << "--final-model options";
    rescore_model_.Read(ki.Stream(), binary);
  }

  if (!config.fmllr_basis_rxfilename.empty()) {
    // We could just as easily use ReadKaldiObject() here.
    bool binary;
    Input ki(config.fmllr_basis_rxfilename, &binary);
    fmllr_basis_.Read(ki.Stream(), binary);
  }
}


const TransitionModel &OnlineGmmDecodingModels::GetTransitionModel() const {
  return tmodel_;
}

const AmDiagGmm &OnlineGmmDecodingModels::GetOnlineAlignmentModel() const {
  if (online_alignment_model_.NumPdfs() != 0)
    return online_alignment_model_;
  else
    return model_;
}

const AmDiagGmm &OnlineGmmDecodingModels::GetModel() const {
  return model_;
}

const AmDiagGmm &OnlineGmmDecodingModels::GetFinalModel() const {
  if (rescore_model_.NumPdfs() != 0)
    return rescore_model_;
  else
    return model_;
}

const BasisFmllrEstimate &OnlineGmmDecodingModels::GetFmllrBasis() const {
  return fmllr_basis_;  
}

}  // namespace kaldi
