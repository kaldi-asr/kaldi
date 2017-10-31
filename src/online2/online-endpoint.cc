// online2/online-endpoint.cc

// Copyright    2014  Johns Hopkins University (author: Daniel Povey)

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

#include "online2/online-endpoint.h"
#include "decoder/lattice-faster-online-decoder.h"

namespace kaldi {

static bool RuleActivated(const OnlineEndpointRule &rule,
                          const std::string &rule_name,
                          BaseFloat trailing_silence,
                          BaseFloat relative_cost,
                          BaseFloat utterance_length) {
  bool contains_nonsilence = (utterance_length > trailing_silence);

  bool ans = (contains_nonsilence || !rule.must_contain_nonsilence) &&
      trailing_silence >= rule.min_trailing_silence &&
      relative_cost <= rule.max_relative_cost &&
      utterance_length >= rule.min_utterance_length;
  if (ans) {
    KALDI_VLOG(2) << "Endpointing rule " << rule_name << " activated: "
                  << (contains_nonsilence ? "true" : "false" ) << ','
                  << trailing_silence << ',' << relative_cost << ','
                  << utterance_length;
  }
  return ans;
}

bool EndpointDetected(const OnlineEndpointConfig &config,
                      int32 num_frames_decoded,
                      int32 trailing_silence_frames,
                      BaseFloat frame_shift_in_seconds,
                      BaseFloat final_relative_cost) {
  KALDI_ASSERT(num_frames_decoded >= trailing_silence_frames);

  BaseFloat utterance_length = num_frames_decoded * frame_shift_in_seconds,
      trailing_silence = trailing_silence_frames * frame_shift_in_seconds;

  if (RuleActivated(config.rule1, "rule1",
                    trailing_silence, final_relative_cost, utterance_length))
    return true;
  if (RuleActivated(config.rule2, "rule2",
                    trailing_silence, final_relative_cost, utterance_length))
    return true;
  if (RuleActivated(config.rule3, "rule3",
                    trailing_silence, final_relative_cost, utterance_length))
    return true;
  if (RuleActivated(config.rule4, "rule4",
                    trailing_silence, final_relative_cost, utterance_length))
    return true;
  if (RuleActivated(config.rule5, "rule5",
                    trailing_silence, final_relative_cost, utterance_length))
    return true;
  return false;
}

int32 TrailingSilenceLength(const TransitionModel &tmodel,
                            const std::string &silence_phones_str,
                            const LatticeFasterOnlineDecoder &decoder) {
  std::vector<int32> silence_phones;
  if (!SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones))
    KALDI_ERR << "Bad --silence-phones option in endpointing config: "
              << silence_phones_str;
  std::sort(silence_phones.begin(), silence_phones.end());
  KALDI_ASSERT(IsSortedAndUniq(silence_phones) &&
               "Duplicates in --silence-phones option in endpointing config");
  KALDI_ASSERT(!silence_phones.empty() &&
               "Endpointing requires nonempty --endpoint.silence-phones option");
  ConstIntegerSet<int32> silence_set(silence_phones);

  bool use_final_probs = false;
  LatticeFasterOnlineDecoder::BestPathIterator iter =
      decoder.BestPathEnd(use_final_probs, NULL);
  int32 num_silence_frames = 0;
  while (!iter.Done()) {  // we're going backwards in time from the most
                          // recently decoded frame...
    LatticeArc arc;
    iter = decoder.TraceBackBestPath(iter, &arc);
    if (arc.ilabel != 0) {
      int32 phone = tmodel.TransitionIdToPhone(arc.ilabel);
      if (silence_set.count(phone) != 0) {
        num_silence_frames++;
      } else {
        break; // stop counting as soon as we hit non-silence.
      }
    }
  }
  return num_silence_frames;
}

bool EndpointDetected(
    const OnlineEndpointConfig &config,
    const TransitionModel &tmodel,
    BaseFloat frame_shift_in_seconds,
    const LatticeFasterOnlineDecoder &decoder) {
  if (decoder.NumFramesDecoded() == 0) return false;

  BaseFloat final_relative_cost = decoder.FinalRelativeCost();

  int32 num_frames_decoded = decoder.NumFramesDecoded(),
      trailing_silence_frames = TrailingSilenceLength(tmodel,
                                                      config.silence_phones,
                                                      decoder);

  return EndpointDetected(config, num_frames_decoded, trailing_silence_frames,
                          frame_shift_in_seconds, final_relative_cost);
}


}  // namespace kaldi
