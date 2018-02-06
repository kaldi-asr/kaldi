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

#include "online2/online-nnet3-cuda-decoding.h"
#include "lat/lattice-functions.h"
#include "lat/determinize-lattice-pruned.h"

namespace kaldi {

SingleUtteranceNnet3CudaDecoder::SingleUtteranceNnet3CudaDecoder(
    const CudaDecoderConfig &decoder_opts,
    const TransitionModel &trans_model,
    const nnet3::DecodableNnetSimpleLoopedInfo &info,
    const CudaFst &fst,
    OnlineNnet2FeaturePipeline *features):
    decoder_opts_(decoder_opts),
    input_feature_frame_shift_in_seconds_(features->FrameShiftInSeconds()),
    trans_model_(trans_model),
    decodable_(trans_model_, info,
               features->InputFeature(), features->IvectorFeature()),
    decoder_(fst, decoder_opts_) {
  decoder_.InitDecoding();
}

void SingleUtteranceNnet3CudaDecoder::AdvanceDecoding() {
  decoder_.AdvanceDecoding(&decodable_);
}

void SingleUtteranceNnet3CudaDecoder::Decode() {
  decoder_.Decode(&decodable_);
}


int32 SingleUtteranceNnet3CudaDecoder::NumFramesDecoded() const {
  return decoder_.NumFramesDecoded();
}

void SingleUtteranceNnet3CudaDecoder::GetBestPath(bool end_of_utterance,
                                              Lattice *best_path) const {
  decoder_.GetBestPath(best_path, end_of_utterance);
}

}  // namespace kaldi
