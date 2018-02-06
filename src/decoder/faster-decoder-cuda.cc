// decoder/faster-decoder-cuda.cc

// Copyright      2018  Zhehuai Chen

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

#include "lat/lattice-functions.h"

#include "cuda-decoder-utils.h"
#include "decoder/faster-decoder-cuda.h"

namespace kaldi {
FasterDecoderCuda::FasterDecoderCuda(const CudaDecoderConfig &decoder_opts,
                              const CudaFst &fst):
    decoder_opts_(decoder_opts), decoder_(fst, decoder_opts_) {
  //decoder_.InitDecoding();
}


void FasterDecoderCuda::Decode(DecodableInterface *decodable) {
  decoder_.Decode(decodable);
  }

// TODO: 
/*
void FasterDecoderCuda::AdvanceDecoding(DecodableInterface *decodable,
                                      int32 max_num_frames) {
  decoder_.AdvanceDecoding(decodable);
}
*/



int32 FasterDecoderCuda::NumFramesDecoded() const {
  return decoder_.NumFramesDecoded();
}

bool FasterDecoderCuda::GetBestPath(Lattice *best_path,
                                    bool use_final_probs) const {
  return decoder_.GetBestPath(best_path, true);
}





} // end namespace kaldi.
