// online2/online-nnet2-decoding.cc

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

#include "online2/online-nnet2-decoding.h"
#include "lat/lattice-functions.h"
#include "lat/determinize-lattice-pruned.h"

namespace kaldi {

SingleUtteranceNnet2Decoder::SingleUtteranceNnet2Decoder(
    const OnlineNnet2DecodingConfig &config,
    const TransitionModel &tmodel,
    const nnet2::AmNnet &model,
    const fst::Fst<fst::StdArc> &fst,
    OnlineFeatureInterface *feature_pipeline):
    config_(config),
    feature_pipeline_(feature_pipeline),
    tmodel_(tmodel),
    decodable_(model, tmodel, config.decodable_opts, feature_pipeline),
    decoder_(fst, config.decoder_opts) {
  decoder_.InitDecoding();
}

void SingleUtteranceNnet2Decoder::AdvanceDecoding() {
  decoder_.AdvanceDecoding(&decodable_);
}

void SingleUtteranceNnet2Decoder::FinalizeDecoding() {
  decoder_.FinalizeDecoding();
}

int32 SingleUtteranceNnet2Decoder::NumFramesDecoded() const {
  return decoder_.NumFramesDecoded();
}

void SingleUtteranceNnet2Decoder::GetLattice(bool end_of_utterance,
                                             CompactLattice *clat) const {
  if (NumFramesDecoded() == 0)
    KALDI_ERR << "You cannot get a lattice if you decoded no frames.";
  Lattice raw_lat;
  decoder_.GetRawLattice(&raw_lat, end_of_utterance);

  if (!config_.decoder_opts.determinize_lattice)
    KALDI_ERR << "--determinize-lattice=false option is not supported at the moment";

  BaseFloat lat_beam = config_.decoder_opts.lattice_beam;
  DeterminizeLatticePhonePrunedWrapper(
      tmodel_, &raw_lat, lat_beam, clat, config_.decoder_opts.det_opts);
}

void SingleUtteranceNnet2Decoder::GetBestPath(bool end_of_utterance,
                                              Lattice *best_path) const {
  decoder_.GetBestPath(best_path, end_of_utterance);
}

bool SingleUtteranceNnet2Decoder::EndpointDetected(
    const OnlineEndpointConfig &config) {
  return kaldi::EndpointDetected(config, tmodel_,
                                 feature_pipeline_->FrameShiftInSeconds(),
                                 decoder_);  
}


}  // namespace kaldi

