// online2/online-nnet3-decoding.cc

// Copyright    2013-2014  Johns Hopkins University (author: Daniel Povey)
//              2016  Api.ai (Author: Ilya Platonov)

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

#include "online2/online-nnet3-decoding.h"
#include "lat/lattice-functions.h"
#include "lat/determinize-lattice-pruned.h"
#include "decoder/grammar-fst.h"

namespace kaldi {

template <typename FST>
SingleUtteranceNnet3DecoderTpl<FST>::SingleUtteranceNnet3DecoderTpl(
    const LatticeFasterDecoderConfig &decoder_opts,
    const TransitionModel &trans_model,
    const nnet3::DecodableNnetSimpleLoopedInfo &info,
    const FST &fst,
    OnlineNnet2FeaturePipeline *features):
    decoder_opts_(decoder_opts),
    input_feature_frame_shift_in_seconds_(features->FrameShiftInSeconds()),
    trans_model_(trans_model),
    decodable_(trans_model_, info,
               features->InputFeature(), features->IvectorFeature()),
    decoder_(fst, decoder_opts_) {
  decoder_.InitDecoding();
}

template <typename FST>
void SingleUtteranceNnet3DecoderTpl<FST>::InitDecoding(int32 frame_offset) {
  decoder_.InitDecoding();
  decodable_.SetFrameOffset(frame_offset);
}

template <typename FST>
void SingleUtteranceNnet3DecoderTpl<FST>::AdvanceDecoding() {
  decoder_.AdvanceDecoding(&decodable_);
}

template <typename FST>
void SingleUtteranceNnet3DecoderTpl<FST>::FinalizeDecoding() {
  decoder_.FinalizeDecoding();
}

template <typename FST>
int32 SingleUtteranceNnet3DecoderTpl<FST>::NumFramesDecoded() const {
  return decoder_.NumFramesDecoded();
}

template <typename FST>
void SingleUtteranceNnet3DecoderTpl<FST>::GetLattice(bool end_of_utterance,
                                             CompactLattice *clat) const {
  if (NumFramesDecoded() == 0)
    KALDI_ERR << "You cannot get a lattice if you decoded no frames.";
  Lattice raw_lat;
  decoder_.GetRawLattice(&raw_lat, end_of_utterance);

  if (!decoder_opts_.determinize_lattice)
    KALDI_ERR << "--determinize-lattice=false option is not supported at the moment";

  BaseFloat lat_beam = decoder_opts_.lattice_beam;
  DeterminizeLatticePhonePrunedWrapper(
      trans_model_, &raw_lat, lat_beam, clat, decoder_opts_.det_opts);
}

template <typename FST>
void SingleUtteranceNnet3DecoderTpl<FST>::GetBestPath(bool end_of_utterance,
                                              Lattice *best_path) const {
  decoder_.GetBestPath(best_path, end_of_utterance);
}

template <typename FST>
bool SingleUtteranceNnet3DecoderTpl<FST>::EndpointDetected(
    const OnlineEndpointConfig &config) {
  BaseFloat output_frame_shift =
      input_feature_frame_shift_in_seconds_ *
      decodable_.FrameSubsamplingFactor();
  return kaldi::EndpointDetected(config, trans_model_,
                                 output_frame_shift, decoder_);
}


// Instantiate the template for the types needed.
template class SingleUtteranceNnet3DecoderTpl<fst::Fst<fst::StdArc> >;
template class SingleUtteranceNnet3DecoderTpl<fst::ConstGrammarFst >;
template class SingleUtteranceNnet3DecoderTpl<fst::VectorGrammarFst >;

}  // namespace kaldi
