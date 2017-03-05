// online2/online-nnet2-decoding.h

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_ONLINE2_ONLINE_NNET2_DECODING_H_
#define KALDI_ONLINE2_ONLINE_NNET2_DECODING_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "nnet2/online-nnet2-decodable.h"
#include "itf/online-feature-itf.h"
#include "online2/online-endpoint.h"
#include "decoder/lattice-faster-online-decoder.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"

namespace kaldi {
/// @addtogroup  onlinedecoding OnlineDecoding
/// @{





// This configuration class contains the configuration classes needed to create
// the class SingleUtteranceNnet2Decoder.  The actual command line program
// requires other configs that it creates separately, and which are not included
// here: namely, OnlineNnet2FeaturePipelineConfig and OnlineEndpointConfig.
struct OnlineNnet2DecodingConfig {
  
  LatticeFasterDecoderConfig decoder_opts;
  nnet2::DecodableNnet2OnlineOptions decodable_opts;
  
  OnlineNnet2DecodingConfig() {  decodable_opts.acoustic_scale = 0.1; }
  
  void Register(OptionsItf *opts) {
    decoder_opts.Register(opts);
    decodable_opts.Register(opts);
  }
};

/**
   You will instantiate this class when you want to decode a single
   utterance using the online-decoding setup for neural nets.
*/
class SingleUtteranceNnet2Decoder {
 public:
  // Constructor.  The feature_pipeline_ pointer is not owned in this
  // class, it's owned externally.
  SingleUtteranceNnet2Decoder(const OnlineNnet2DecodingConfig &config,
                              const TransitionModel &tmodel,
                              const nnet2::AmNnet &model,
                              const fst::Fst<fst::StdArc> &fst,
                              OnlineFeatureInterface *feature_pipeline);
  
  /// advance the decoding as far as we can.
  void AdvanceDecoding();

  /// Finalizes the decoding. Cleans up and prunes remaining tokens, so the
  /// GetLattice() call will return faster.  You must not call this before
  /// calling (TerminateDecoding() or InputIsFinished()) and then Wait().
  void FinalizeDecoding();

  int32 NumFramesDecoded() const;
  
  /// Gets the lattice.  The output lattice has any acoustic scaling in it
  /// (which will typically be desirable in an online-decoding context); if you
  /// want an un-scaled lattice, scale it using ScaleLattice() with the inverse
  /// of the acoustic weight.  "end_of_utterance" will be true if you want the
  /// final-probs to be included.
  void GetLattice(bool end_of_utterance,
                  CompactLattice *clat) const;
  
  /// Outputs an FST corresponding to the single best path through the current
  /// lattice. If "use_final_probs" is true AND we reached the final-state of
  /// the graph then it will include those as final-probs, else it will treat
  /// all final-probs as one.
  void GetBestPath(bool end_of_utterance,
                   Lattice *best_path) const;


  /// This function calls EndpointDetected from online-endpoint.h,
  /// with the required arguments.
  bool EndpointDetected(const OnlineEndpointConfig &config);

  const LatticeFasterOnlineDecoder &Decoder() const { return decoder_; }
  
  ~SingleUtteranceNnet2Decoder() { }
 private:

  OnlineNnet2DecodingConfig config_;

  OnlineFeatureInterface *feature_pipeline_;

  const TransitionModel &tmodel_;
  
  nnet2::DecodableNnet2Online decodable_;
  
  LatticeFasterOnlineDecoder decoder_;
  
};

  
/// @} End of "addtogroup onlinedecoding"

}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_NNET2_DECODING_H_
