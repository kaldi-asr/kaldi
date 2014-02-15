// online2/online-gmm-decoding.cc

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

#include "online2/online-gmm-decoding.h"
#include "lat/lattice-functions.h"

namespace kaldi {


SingleUtteranceGmmDecoder::SingleUtteranceGmmDecoder(
    const OnlineGmmDecodingConfig &config,
    const OnlineGmmDecodingModels &models,                            
    const OnlineFeaturePipeline &feature_prototype,
    const fst::Fst<fst::StdArc> &fst,
    const SpeakerAdaptationState &adaptation_state):
    config_(config), models_(models),
    feature_pipeline_(feature_prototype.Copy()),
    orig_adaptation_state_(adaptation_state),
    adaptation_state_(adaptation_state),
    decoder_(fst, config.faster_decoder_opts) {
  feature_pipeline_->SetTransform(adaptation_state_.cur_transform);
}
    

/** Advance the first-pass decoding as far as we can. */
void SingleUtteranceGmmDecoder::AdvanceFirstPass() {
  bool have_transform = HasTransform();
  
  // have_transform is true if we already have a transform set up.  This affects
  // whether we use, as a first choice, the SAT-trained model or the
  // speaker-independent model.  [If the user supplied only one model we'll use
  // that one, of course].

  const AmDiagGmm &am_gmm = (have_transform ? models_.GetModel() :
                             models_.GetOnlineAlignmentModel());

  // The decodable object is lightweight, we lose nothing
  // from constructing it each time we want to decode more of the
  // input.
  DecodableDiagGmmScaledOnline decodable(am_gmm,
                                         models_.GetTransitionModel(),
                                         config_.acoustic_scale,
                                         feature_pipeline_);

  // This will decode as many frames as are currently available.
  decoder_.Decode(&decodable);  
}

void SingleUtteranceGmmDecoder::EstimateFmllr(bool end_of_utterance) {
  if (decoder_.NumFramesDecoded() == 0) {
    KALDI_WARN << "You have decoded no data so cannot estimate fMLLR.";
  }
  // Note: we'll just use whatever acoustic scaling factor we were decoding
  // with.  This is in the lattice that we get from decoder_.GetRawLattice().
  Lattice raw_lat;
  decoder_.GetRawLattice(&raw_lat, end_of_utterance);
  KALDI_ASSERT(config_.fmllr_lattice_beam > 0.0);
  PruneLattice(config_.fmllr_lattice_beam, &raw_lat);
  CompactLattice lat; // determinized lattice.
  DeterminizeLatticePruned(raw_lat, config_.fmllr_lattice_beam, &lat);
  if (lat.NumStates() == 0) {
    // Do nothing if the lattice is empty.  This should not happen.
    KALDI_WARN << "Got empty lattice.  Not estimating fMLLR.";
  }

  if (adaptation_state_.spk_stats.beta_ !=
      orig_adaptation_state_.spk_stats.beta_) {
    // This could happen if the user called EstimateFmllr() twice on the
    // same utterance... we don't want to count any stats twice so we
    // have to reset the stats to what they were before this utterance
    // (possibly empty).
    adaptation_state_.spk_stats = orig_adaptation_state_.spk_stats;
  }
  if (adaptation_state_.spk_stats.Dim() == 0)
    adaptation_state_.spk_stats.Init(feature_pipeline_->Dim()); 
    
  
}

void SingleUtteranceGmmDecoder::AccumulateFmllrStats(
    const CompactLattice &clat,
    FmllrDiagGmmAccs *spk_stats) {
}



bool  SingleUtteranceGmmDecoder::HasTransform() const {
  return (feature_pipeline_->GetTransform().NumRows() > 0);
}

}  // namespace kaldi
