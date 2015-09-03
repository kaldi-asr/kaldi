// online/online-faster-decoder.h

// Copyright 2012 Cisco Systems (author: Matthias Paulik)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

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

#ifndef KALDI_ONLINE_ONLINE_FASTER_DECODER_H_
#define KALDI_ONLINE_ONLINE_FASTER_DECODER_H_

#include "util/stl-utils.h"
#include "decoder/faster-decoder.h"
#include "hmm/transition-model.h"

namespace kaldi {

// Extends the definition of FasterDecoder's options to include additional
// parameters. The meaning of the "beam" option is also redefined as
// the _maximum_ beam value allowed.
struct OnlineFasterDecoderOpts : public FasterDecoderOptions {
  BaseFloat rt_min; // minimum decoding runtime factor
  BaseFloat rt_max; // maximum decoding runtime factor
  int32 batch_size; // number of features decoded in one go
  int32 inter_utt_sil; // minimum silence (#frames) to trigger end of utterance
  int32 max_utt_len_; // if utt. is longer, we accept shorter silence as utt. separators
  int32 update_interval; // beam update period in # of frames
  BaseFloat beam_update; // rate of adjustment of the beam
  BaseFloat max_beam_update; // maximum rate of beam adjustment

  OnlineFasterDecoderOpts() :
    rt_min(.7), rt_max(.75), batch_size(27),
    inter_utt_sil(50), max_utt_len_(1500),
    update_interval(3), beam_update(.01),
    max_beam_update(0.05) {}

  void Register(OptionsItf *opts, bool full) {
    FasterDecoderOptions::Register(opts, full);
    opts->Register("rt-min", &rt_min,
                   "Approximate minimum decoding run time factor");
    opts->Register("rt-max", &rt_max,
                   "Approximate maximum decoding run time factor");
    opts->Register("update-interval", &update_interval,
                   "Beam update interval in frames");
    opts->Register("beam-update", &beam_update, "Beam update rate");
    opts->Register("max-beam-update", &max_beam_update, "Max beam update rate");
    opts->Register("inter-utt-sil", &inter_utt_sil,
                   "Maximum # of silence frames to trigger new utterance");
    opts->Register("max-utt-length", &max_utt_len_,
                   "If the utterance becomes longer than this number of frames, "
                   "shorter silence is acceptable as an utterance separator");
  }
};

class OnlineFasterDecoder : public FasterDecoder {
 public:
  // Codes returned by Decode() to show the current state of the decoder
  enum DecodeState {
    kEndFeats = 1, // No more scores are available from the Decodable
    kEndUtt = 2, // End of utterance, caused by e.g. a sufficiently long silence
    kEndBatch = 4 // End of batch - end of utterance not reached yet
  };

  // "sil_phones" - the IDs of all silence phones
  OnlineFasterDecoder(const fst::Fst<fst::StdArc> &fst,
                      const OnlineFasterDecoderOpts &opts,
                      const std::vector<int32> &sil_phones,
                      const TransitionModel &trans_model)
      : FasterDecoder(fst, opts), opts_(opts),
        silence_set_(sil_phones), trans_model_(trans_model),
        max_beam_(opts.beam), effective_beam_(FasterDecoder::config_.beam),
        state_(kEndFeats), frame_(0), utt_frames_(0) {}

  DecodeState Decode(DecodableInterface *decodable);
  
  // Makes a linear graph, by tracing back from the last "immortal" token
  // to the previous one
  bool PartialTraceback(fst::MutableFst<LatticeArc> *out_fst);

  // Makes a linear graph, by tracing back from the best currently active token
  // to the last immortal token. This method is meant to be invoked at the end
  // of an utterance in order to get the last chunk of the hypothesis
  void FinishTraceBack(fst::MutableFst<LatticeArc> *fst_out);

  // Returns "true" if the best current hypothesis ends with long enough silence
  bool EndOfUtterance();

  int32 frame() { return frame_; }

 private:
  void ResetDecoder(bool full);

  // Returns a linear fst by tracing back the last N frames, beginning
  // from the best current token
  void TracebackNFrames(int32 nframes, fst::MutableFst<LatticeArc> *out_fst);

  // Makes a linear "lattice", by tracing back a path delimited by two tokens
  void MakeLattice(const Token *start,
                   const Token *end,
                   fst::MutableFst<LatticeArc> *out_fst) const;

  // Searches for the last token, ancestor of all currently active tokens
  void UpdateImmortalToken();

  const OnlineFasterDecoderOpts opts_;
  const ConstIntegerSet<int32> silence_set_; // silence phones IDs
  const TransitionModel &trans_model_; // needed for trans-id -> phone conversion
  const BaseFloat max_beam_; // the maximum allowed beam
  BaseFloat &effective_beam_; // the currently used beam
  DecodeState state_; // the current state of the decoder
  int32 frame_; // the next frame to be processed
  int32 utt_frames_; // # frames processed from the current utterance
  Token *immortal_tok_;      // "immortal" token means it's an ancestor of ...
  Token *prev_immortal_tok_; // ... all currently active tokens
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineFasterDecoder);
};

} // namespace kaldi
#endif // KALDI_ONLINE_ONLINE_FASTER_DECODER_H_
