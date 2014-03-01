// online2/online-endpoing.cc

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

namespace kaldi {

static inline void GetBestPathStats(const TransitionModel &tmodel,
                                    const Lattice &best_path,
                                    const OnlineEndpointConfig &config,
                                    BaseFloat frame_shift_in_seconds,
                                    BaseFloat *utterance_length,
                                    BaseFloat *trailing_silence) {
  std::vector<int32> silence_phones;
  if (!SplitStringToIntegers(config.silence_phones, ":", false, &silence_phones))
    KALDI_ERR << "Bad --silence-phones option in endpointing config: "
              << config.silence_phones;
  std::sort(silence_phones.begin(), silence_phones.end());
  KALDI_ASSERT(IsSortedAndUniq(silence_phones) &&
               "Duplicates in --silence-phones option in endpointing config");
  KALDI_ASSERT(!silence_phones.empty() &&
               "Endpointing requires nonempty --endpoint.silence-phones option");
  ConstIntegerSet<int32> silence_set(silence_phones);
  
  std::vector<int32> alignment;

  fst::GetLinearSymbolSequence<LatticeArc,int32>(best_path, &alignment, NULL, NULL);

  *utterance_length = alignment.size() * frame_shift_in_seconds;
  if (alignment.empty()) {
    *trailing_silence = 0.0;
    return;
  }
  
  int32 num_sil_frames = 0;
  for (int32 n = alignment.size() - 1; n >= 0; n--) {
    int32 phone = tmodel.TransitionIdToPhone(alignment[n]);
    if (silence_set.count(phone) != 0)
      num_sil_frames++;
    else
      break;
  }
  *trailing_silence = num_sil_frames * frame_shift_in_seconds;
}


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

bool EndpointDetected(const TransitionModel &tmodel,
                      const Lattice &best_path,
                      const OnlineEndpointConfig &config,
                      BaseFloat frame_shift_in_seconds,
                      BaseFloat final_relative_cost) {
  if (best_path.NumStates() == 0) return false;
  KALDI_ASSERT(final_relative_cost >= 0.0);
  BaseFloat utterance_length, trailing_silence;
  GetBestPathStats(tmodel, best_path, config, frame_shift_in_seconds,
                   &utterance_length, &trailing_silence);
  
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
