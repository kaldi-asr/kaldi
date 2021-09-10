// online2/online-endpoint.h

// Copyright 2013   Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_ONLINE2_ONLINE_ENDPOINT_H_
#define KALDI_ONLINE2_ONLINE_ENDPOINT_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "itf/transition-information.h"
#include "lat/kaldi-lattice.h"
#include "decoder/lattice-faster-online-decoder.h"
#include "decoder/lattice-incremental-online-decoder.h"

namespace kaldi {
/// @addtogroup  onlinedecoding OnlineDecoding
/// @{


/**
   This header contains a simple facility for endpointing, that should be used
   in conjunction with the "online2" online decoding code; see
   ../online2bin/online2-wav-gmm-latgen-faster-endpoint.cc.  By endpointing in
   this context we mean "deciding when to stop decoding", and not generic
   speech/silence segmentation.  The use-case that we have in mind is some kind
   of dialog system where, as more speech data comes in, we decode more and
   more, and we have to decide when to stop decoding.

   The endpointing rule is a disjunction of conjunctions.  The way we have
   it configured, it's an OR of five rules, and each rule has the following form:

      (<contains-nonsilence> || !rule.must_contain_nonsilence) &&
       <length-of-trailing-silence> >= rule.min_trailing_silence &&
       <relative-cost> <= rule.max_relative_cost &&
       <utterance-length> >= rule.min_utterance_length)

   where:
    <contains-nonsilence> is true if the best traceback contains any nonsilence phone;
    <length-of-trailing-silence> is the length in seconds of silence phones at the
      end of the best traceback (we stop counting when we hit non-silence),
    <relative-cost> is a value >= 0 extracted from the decoder, that is zero if
      a final-state of the grammar FST had the best cost at the final frame, and
      infinity if no final-state was active (and >0 for in-between cases).
    <utterance-length> is the number of seconds of the utterance that we have
      decoded so far.

   All of these pieces of information are obtained from the best-path
   traceback from the decoder, which is output by the function GetBestPath().
   We do this every time we're finished processing a chunk of data.
   [ Note: we're changing the decoders so that GetBestPath() is efficient
   and does not require operations on the entire lattice. ]

   For details of the default rules, see struct OnlineEndpointConfig.

   It's up to the caller whether to use final-probs or not when generating the
   best-path, i.e. whether to call decoder.GetBestPath(&lat, (true or false)),
   but my recommendation is not to use them.  If you do use them, then depending
   on the grammar, you may force the best-path to decode non-silence even though
   that was not what it really preferred to decode.
 */


struct OnlineEndpointRule {
  bool must_contain_nonsilence;
  BaseFloat min_trailing_silence;
  BaseFloat max_relative_cost;
  BaseFloat min_utterance_length;
  // The values set in the initializer will probably never be used.
  OnlineEndpointRule(bool must_contain_nonsilence = true,
                     BaseFloat min_trailing_silence = 1.0,
                     BaseFloat max_relative_cost = std::numeric_limits<BaseFloat>::infinity(),
                     BaseFloat min_utterance_length = 0.0):
      must_contain_nonsilence(must_contain_nonsilence),
      min_trailing_silence(min_trailing_silence),
      max_relative_cost(max_relative_cost),
      min_utterance_length(min_utterance_length) { }

  void Register(OptionsItf *opts) {
    opts->Register("must-contain-nonsilence", &must_contain_nonsilence,
                   "If true, for this endpointing rule to apply there must "
                   "be nonsilence in the best-path traceback.");
    opts->Register("min-trailing-silence", &min_trailing_silence,
                   "This endpointing rule requires duration of trailing silence"
                   "(in seconds) to be >= this value.");
    opts->Register("max-relative-cost", &max_relative_cost,
                   "This endpointing rule requires relative-cost of final-states"
                   " to be <= this value (describes how good the probability "
                   "of final-states is).");
    opts->Register("min-utterance-length", &min_utterance_length,
                   "This endpointing rule requires utterance-length (in seconds) "
                   "to be >= this value.");
  };
  // for convenience add this RegisterWithPrefix function, because
  // we'll be registering this as a config with several different
  // prefixes.
  void RegisterWithPrefix(const std::string &prefix, OptionsItf *opts) {
    ParseOptions po_prefix(prefix, opts);
    this->Register(&po_prefix);
  }
};

struct OnlineEndpointConfig {
  std::string silence_phones; /// e.g. 1:2:3:4, colon separated list of phones
                              /// that we consider as silence for purposes of
                              /// endpointing.

  /// We support five rules.  We terminate decoding if ANY of these rules
  /// evaluates to "true". If you want to add more rules, do it by changing this
  /// code.  If you want to disable a rule, you can set the silence-timeout for
  /// that rule to a very large number.

  /// rule1 times out after 5 seconds of silence, even if we decoded nothing.
  OnlineEndpointRule rule1;
  /// rule2 times out after 0.5 seconds of silence if we reached the final-state
  /// with good probability (relative_cost < 2.0) after decoding something.
  OnlineEndpointRule rule2;
  /// rule3 times out after 1.0 seconds of silence if we reached the final-state
  /// with OK probability (relative_cost < 8.0) after decoding something
  OnlineEndpointRule rule3;
  /// rule4 times out after 2.0 seconds of silence after decoding something,
  /// even if we did not reach a final-state at all.
  OnlineEndpointRule rule4;
  /// rule5 times out after the utterance is 20 seconds long, regardless of
  /// anything else.
  OnlineEndpointRule rule5;

  OnlineEndpointConfig():
      rule1(false, 5.0, std::numeric_limits<BaseFloat>::infinity(), 0.0),
      rule2(true, 0.5, 2.0, 0.0),
      rule3(true, 1.0, 8.0, 0.0),
      rule4(true, 2.0, std::numeric_limits<BaseFloat>::infinity(), 0.0),
      rule5(false, 0.0, std::numeric_limits<BaseFloat>::infinity(), 20.0) { }

  void Register(OptionsItf *opts) {
    opts->Register("endpoint.silence-phones", &silence_phones, "List of phones "
                   "that are considered to be silence phones by the "
                   "endpointing code.");
    rule1.RegisterWithPrefix("endpoint.rule1", opts);
    rule2.RegisterWithPrefix("endpoint.rule2", opts);
    rule3.RegisterWithPrefix("endpoint.rule3", opts);
    rule4.RegisterWithPrefix("endpoint.rule4", opts);
    rule5.RegisterWithPrefix("endpoint.rule5", opts);
  }
};






/// This function returns true if this set of endpointing
/// rules thinks we should terminate decoding.  Note: in verbose
/// mode it will print logging information when returning true.
bool EndpointDetected(const OnlineEndpointConfig &config,
                      int32 num_frames_decoded,
                      int32 trailing_silence_frames,
                      BaseFloat frame_shift_in_seconds,
                      BaseFloat final_relative_cost);


/// returns the number of frames of trailing silence in the best-path traceback
/// (not using final-probs).  "silence_phones" is a colon-separated list of
/// integer id's of phones that we consider silence.  We use the the
/// BestPathEnd() and TraceBackOneLink() functions of LatticeFasterOnlineDecoder
/// to do this efficiently.
template <typename DEC>
int32 TrailingSilenceLength(const TransitionInformation &tmodel,
                            const std::string &silence_phones,
                            const DEC &decoder);


/// This is a higher-level convenience function that works out the
/// arguments to the EndpointDetected function above, from the decoder.
template <typename DEC>
bool EndpointDetected(
    const OnlineEndpointConfig &config,
    const TransitionInformation &tmodel,
    BaseFloat frame_shift_in_seconds,
    const DEC &decoder);



/// @} End of "addtogroup onlinedecoding"

}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_ENDPOINT_
