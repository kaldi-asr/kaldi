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

#include "cudadecoder/online-endpoint.h"

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
}  // namespace kaldi
