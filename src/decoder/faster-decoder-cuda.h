// decoder/faster-decoder-cuda.h

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

#ifndef KALDI_DECODER_FASTER_DECODER_CUDA_H_
#define KALDI_DECODER_FASTER_DECODER_CUDA_H_

#include "util/stl-utils.h"
#include "itf/options-itf.h"
#include "util/hash-list.h"
#include "cudamatrix/cu-common.h"
#include "itf/decodable-itf.h"

#include "decoder/cuda-decoder.h"

namespace kaldi {

class FasterDecoderCuda {
 public:

  FasterDecoderCuda(const CudaDecoderConfig &decoder_opts,
                const CudaFst &fst);

  ~FasterDecoderCuda() { }

  const CudaDecoder &Decoder() const { return decoder_; }
  
  void Decode(DecodableInterface *decodable);

  /// GetBestPath gets the decoding traceback. If "use_final_probs" is true
  /// AND we reached a final state, it limits itself to final states;
  /// otherwise it gets the most likely token not taking into account
  /// final-probs. Returns true if the output best path was not the empty
  /// FST (will only return false in unusual circumstances where
  /// no tokens survived).
  bool GetBestPath(Lattice *best_path,
                   bool use_final_probs = true) const;

  bool ReachedFinal() const { return decoder_.ReachedFinal(); }
  /// As a new alternative to Decode(), you can call InitDecoding
  /// and then (possibly multiple times) AdvanceDecoding().
  void InitDecoding();


  /// This will decode until there are no more frames ready in the decodable
  /// object, but if max_num_frames is >= 0 it will decode no more than
  /// that many frames.
  // TODO
  //void AdvanceDecoding(DecodableInterface *decodable,
  //                     int32 max_num_frames = -1);

  /// Returns the number of frames already decoded.
  int32 NumFramesDecoded() const;

 protected:

  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;

 private:

  const CudaDecoderConfig &decoder_opts_;
  CudaDecoder decoder_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(FasterDecoderCuda);
};


} // end namespace kaldi.


#endif
