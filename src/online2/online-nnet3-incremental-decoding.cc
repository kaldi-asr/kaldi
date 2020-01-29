// online2/online-nnet3-incremental-decoding.cc

// Copyright      2019  Zhehuai Chen

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

#include "online2/online-nnet3-incremental-decoding.h"
#include "lat/lattice-functions.h"
#include "lat/determinize-lattice-pruned.h"
#include "decoder/grammar-fst.h"

namespace kaldi {

template <typename FST>
SingleUtteranceNnet3IncrementalDecoderTpl<FST>::SingleUtteranceNnet3IncrementalDecoderTpl(
    const LatticeIncrementalDecoderConfig &decoder_opts,
    const TransitionModel &trans_model,
    const nnet3::DecodableNnetSimpleLoopedInfo &info,
    const FST &fst,
    OnlineNnet2FeaturePipeline *features):
    decoder_opts_(decoder_opts),
    input_feature_frame_shift_in_seconds_(features->FrameShiftInSeconds()),
    trans_model_(trans_model),
    decodable_(trans_model_, info,
               features->InputFeature(), features->IvectorFeature()),
    decoder_(fst, trans_model, decoder_opts_) {
  decoder_.InitDecoding();
}

template <typename FST>
void SingleUtteranceNnet3IncrementalDecoderTpl<FST>::InitDecoding(int32 frame_offset) {
  decoder_.InitDecoding();
  decodable_.SetFrameOffset(frame_offset);
}

template <typename FST>
void SingleUtteranceNnet3IncrementalDecoderTpl<FST>::AdvanceDecoding() {
  decoder_.AdvanceDecoding(&decodable_);
}

template <typename FST>
void SingleUtteranceNnet3IncrementalDecoderTpl<FST>::GetBestPath(bool end_of_utterance,
                                              Lattice *best_path) const {
  decoder_.GetBestPath(best_path, end_of_utterance);
}

template <typename FST>
bool SingleUtteranceNnet3IncrementalDecoderTpl<FST>::EndpointDetected(
    const OnlineEndpointConfig &config) {
  BaseFloat output_frame_shift =
      input_feature_frame_shift_in_seconds_ *
      decodable_.FrameSubsamplingFactor();
  return kaldi::EndpointDetected(config, trans_model_,
                                 output_frame_shift, decoder_);
}


// Instantiate the template for the types needed.
template class SingleUtteranceNnet3IncrementalDecoderTpl<fst::Fst<fst::StdArc> >;
template class SingleUtteranceNnet3IncrementalDecoderTpl<fst::GrammarFst>;

}  // namespace kaldi
