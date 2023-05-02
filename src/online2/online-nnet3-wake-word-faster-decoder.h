// online2/online-nnet3-wake-word-faster-decoder.h

// Copyright 2019-2020  Daniel Povey
//           2019-2020  Yiming Wang


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

#ifndef KALDI_ONLINE2_ONLINE_NNET3_WAKE_WORD_FASTER_DECODER_H_
#define KALDI_ONLINE2_ONLINE_NNET3_WAKE_WORD_FASTER_DECODER_H_

#include "util/stl-utils.h"
#include "decoder/faster-decoder.h"
#include "itf/online-feature-itf.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "hmm/transition-model.h"

namespace kaldi {

/// Extends the FasterDecoder's options with additional parameters
/// (currently none).
struct OnlineWakeWordFasterDecoderOpts : public FasterDecoderOptions {
  void Register(OptionsItf *opts, bool full) {
    FasterDecoderOptions::Register(opts, full);
  }
};

/** This is code is modified from online/online-faster-decoder.h and
    online2/online-nnet3-decoding.h for nnet3 online decoding in wake word
    detection. It uses `immortal tokens` from OnlineFasterDecoder for patial
    tracing back to obtain partial hypotheses while decoding a recording.
    Different from OnlineFasterDecoder, tt doesn't have end-point detection,
    and doesn't use run-time factor to adjust the beam.
*/

class OnlineWakeWordFasterDecoder : public FasterDecoder {
 public:
  OnlineWakeWordFasterDecoder(const fst::Fst<fst::StdArc> &fst,
                              const OnlineWakeWordFasterDecoderOpts &opts)
      : FasterDecoder(fst, opts), opts_(opts) {}

  // Makes a linear graph, by tracing back from the last "immortal" token
  // to the previous one
  bool PartialTraceback(fst::MutableFst<LatticeArc> *out_fst);

  // Makes a linear graph, by tracing back from the best currently active token
  // to the last immortal token. This method is meant to be invoked at the end
  // of an utterance in order to get the last chunk of the hypothesis
  void FinishTraceBack(fst::MutableFst<LatticeArc> *fst_out);

  // As a new alternative to Decode(), you can call InitDecoding
  // and then (possibly multiple times) AdvanceDecoding().
  void InitDecoding();

 private:
  // Returns a linear fst by tracing back the last N frames, beginning
  // from the best current token
  void TracebackNFrames(int32 nframes, fst::MutableFst<LatticeArc> *out_fst);

  // Makes a linear "lattice", by tracing back a path delimited by two tokens
  void MakeLattice(const Token *start,
                   const Token *end,
                   fst::MutableFst<LatticeArc> *out_fst) const;

  // Searches for the last token, ancestor of all currently active tokens
  void UpdateImmortalToken();

  const OnlineWakeWordFasterDecoderOpts opts_;
  Token *immortal_tok_;      // "immortal" token means it's an ancestor of ...
  Token *prev_immortal_tok_; // ... all currently active tokens
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineWakeWordFasterDecoder);
};

} // namespace kaldi
#endif // KALDI_ONLINE2_ONLINE_NNET3_WAKE_WORD_FASTER_DECODER_H_
