// chain/language-model.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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

#include <algorithm>
#include <numeric>
#include "chain/language-model.h"
#include "util/simple-io-funcs.h"


namespace kaldi {
namespace chain {

void LanguageModelEstimator::AddCounts(const std::vector<int32> &sentence) {
  int32 order = opts_.ngram_order;
  KALDI_ASSERT(order > 0);
  // 0 is used for left-context at the beginning of the file.. treat it as BOS.
  std::vector<int32> history(order - 1, 0);
  std::vector<int32>::const_iterator iter = sentence.begin(),
      end = sentence.end();
  int32 max_phone = 0;
  for (; iter != end; ++iter) {
    KALDI_ASSERT(*iter != 0);
    history.push_back(*iter);
    max_phone = std::max(max_phone, *iter);
    IncrementCount(history);
    history.erase(history.begin());
  }
  max_phone_ = std::max(max_phone_, max_phone);
  // Probability of end of sentence.  This will end up getting ignored later.
  history.push_back(0);
  IncrementCount(history);
}

void LanguageModelEstimator::IncrementCount(const std::vector<int32> &ngram) {
  // we could do this more efficiently but this is probably OK for now.
  MapType::iterator iter = counts_.find(ngram);
  if (iter != counts_.end())
    iter->second++;
  else
    counts_[ngram] = 1;
}


void LanguageModelEstimator::ComputePhoneSets(
    std::vector<int32> *phone_to_set) const {
  KALDI_ASSERT(max_phone_ > 0 && "Saw no data.");
  phone_to_set->resize(max_phone_ + 1);
  (*phone_to_set)[0] = 0;
  if (opts_.leftmost_context_questions_rxfilename.empty()) {
    for (int32 p = 1; p <= max_phone_; p++)
      (*phone_to_set)[p] = p;
  } else {

    std::vector<std::vector<int32> > questions;  // sets of phones.
    ReadIntegerVectorVectorSimple(opts_.leftmost_context_questions_rxfilename,
                                  &questions);
    for (size_t i = 0; i < questions.size(); i++) {
      SortAndUniq(&(questions[i]));
    }

    // note, by 'set' here we mean the integer identifier assigned to
    // a set of phones that is split the same way by the questions.
    int32 cur_set = 0;
    std::map<vector<bool>, int32> answers_to_set;
    std::map<vector<bool>, int32>::iterator iter;
    for (int32 p = 1; p <= max_phone_; p++) {
      std::vector<bool> answers(questions.size());
      for (int32 i = 0; i < questions.size(); i++)
        answers[i] = std::binary_search(questions[i].begin(),
                                        questions[i].end(), p);
      if ((iter = answers_to_set.find(answers)) == answers_to_set.end()) {
        cur_set++;
        (*phone_to_set)[p] = cur_set;
        answers_to_set[answers] = cur_set;
      } else {
        (*phone_to_set)[p] = iter->second;
      }
    }
    KALDI_LOG << "Reduced " << (max_phone_ + 1) << " phones to "
              << cur_set << " sets (when appearing in left-most position)";
  }
}

void LanguageModelEstimator::Estimate(fst::StdVectorFst *fst) const {

  std::vector<int32> phone_to_set;
  ComputePhoneSets(&phone_to_set);
  MapType hist_counts;
  GetHistoryCounts(&hist_counts);
  int32 count_cutoff = GetHistoryCountCutoff(hist_counts);
  MapType hist_to_state;
  int32 num_history_states = GetHistoryToStateMap(hist_counts, count_cutoff,
                                                  phone_to_set, &hist_to_state);
  { MapType temp; temp.swap(hist_counts); }  // this won't be needed from this
                                             // point; free memory.
  PairMapType transitions;
  int32 initial_state = GetStateTransitions(hist_to_state, &transitions);
  PairMapType num_counts;
  std::vector<int32> den_counts;
  GetCounts(hist_to_state, num_history_states, &num_counts, &den_counts);

  OutputToFst(initial_state, num_counts, den_counts, transitions, fst);
}

void LanguageModelEstimator::GetCounts(
    const LanguageModelEstimator::MapType &hist_to_state,
    int32 num_history_states,
    LanguageModelEstimator::PairMapType *num_counts,
    std::vector<int32> *den_counts) const {
  den_counts->clear();
  den_counts->resize(num_history_states, 0);
  MapType::const_iterator iter = counts_.begin(), end = counts_.end();
  for (; iter != end; ++iter) {
    std::vector<int32> hist = iter->first;  // at this point it's an ngram.
    int32 count = iter->second;
    int32 phone = hist.back();
    hist.pop_back();  // now it's a history.
    MapType::const_iterator hist_to_state_iter = hist_to_state.find(hist);
    KALDI_ASSERT(hist_to_state_iter != hist_to_state.end());
    int32 hist_state = hist_to_state_iter->second;
    den_counts->at(hist_state) += count;
    std::pair<int32,int32> p(hist_state, phone);
    PairMapType::iterator iter = num_counts->find(p);
    if (iter == num_counts->end()) {
      (*num_counts)[p] = count;
    } else {
      iter->second += count;
    }
  }
}

int32 LanguageModelEstimator::GetHistoryCountCutoff(
    const LanguageModelEstimator::MapType &hist_counts) const {
  int32 ans;
  if (opts_.num_extra_states == 0 ||
      opts_.leftmost_context_questions_rxfilename.empty()) {
    ans = std::numeric_limits<int32>::max();
  } else if (hist_counts.size() <= opts_.num_extra_states) {
    ans = 0;
  } else {
    std::vector<int32> counts;
    counts.reserve(hist_counts.size());
    for (MapType::const_iterator iter = hist_counts.begin(),
             end = hist_counts.end(); iter != end; ++iter) {
      int32 count = iter->second;
      counts.push_back(count);
    }
    std::vector<int32>::iterator mid = counts.end() -
        opts_.num_extra_states + 1;
    std::nth_element(counts.begin(), mid, counts.end());
    ans = *mid;
  }
  KALDI_LOG << "For --ngram-order=" << opts_.ngram_order
            << ", --num-extra-states=" << opts_.num_extra_states
            << " and --leftmost-context-uestions='"
            << opts_.leftmost_context_questions_rxfilename
            << "', count cutoff to control state merging is "
            << ans;
  return ans;
}

int32 LanguageModelEstimator::GetHistoryToStateMap(
    const LanguageModelEstimator::MapType &hist_counts,
    int32 count_cutoff,
    const std::vector<int32> &phone_to_set,
    LanguageModelEstimator::MapType *hist_to_state) const {
  if (opts_.ngram_order == 1) {
    // special case for order = 1.
    (*hist_to_state)[ std::vector<int32>() ] = 0;
    return 1;
  } else {
    // mapped_hist_to_state maps the history *after mapping its leftmost phone
    // to its equivalence class* to the state.
    MapType mapped_hist_to_state;
    int32 num_states = 0;
    MapType::const_iterator iter = hist_counts.begin(),
        end = hist_counts.end();
    for (; iter != end; ++iter) {
      const std::vector<int32> &hist = iter->first;
      int32 count = iter->second,
          this_state;
      if (count >= count_cutoff) {
        // give it its own state.
        KALDI_ASSERT((*hist_to_state).find(hist) == hist_to_state->end());
        this_state = num_states++;
      } else {
        // map the leftmost phone to the equivalence class...
        std::vector<int32> mapped_hist(hist);
        mapped_hist[0] = phone_to_set.at(mapped_hist[0]);
        // and see if we already have a state allocated for this
        // equivalence-class of histories.
        MapType::const_iterator map_iter =
            mapped_hist_to_state.find(mapped_hist);
        if (map_iter == mapped_hist_to_state.end()) {
          this_state = num_states++;
          mapped_hist_to_state[mapped_hist] = this_state;
        } else {
          this_state = map_iter->second;
        }
      }
      (*hist_to_state)[hist] = this_state;
    }
    return num_states;
  }
}


void LanguageModelEstimator::GetHistoryCounts(MapType *hist_counts) const {
  hist_counts->clear();
  int32 num_ngrams = 0;
  int64 tot_count = 0;
  MapType::const_iterator counts_iter = counts_.begin(),
      counts_end = counts_.end();
  for (; counts_iter != counts_end; ++counts_iter) {
    std::vector<int32> this_hist(counts_iter->first);
    this_hist.pop_back();
    int32 count = counts_iter->second;
    MapType::iterator iter = hist_counts->find(this_hist);
    if (iter == hist_counts->end())
      (*hist_counts)[this_hist] = count;
    else
      iter->second += count;
    num_ngrams++;
    tot_count += count;
  }
  KALDI_LOG << "Saw " << tot_count << " phones, " << num_ngrams
            << " unique ngrams and " << hist_counts->size()
            << " unique history states.";
}

int32 LanguageModelEstimator::GetStateTransitions(
    const MapType &hist_to_state,
    LanguageModelEstimator::PairMapType *transitions) const {
  MapType::const_iterator counts_iter = counts_.begin(),
      counts_end = counts_.end();
  for (; counts_iter != counts_end; ++counts_iter) {
    std::vector<int32> this_hist(counts_iter->first),
        next_hist(counts_iter->first);
    this_hist.pop_back();
    int32 phone = next_hist.back();
    next_hist.erase(next_hist.begin());
    if (phone == 0)
      continue;
    int32 this_state, next_state;
    MapType::const_iterator hist_to_state_iter = hist_to_state.find(this_hist);
    KALDI_ASSERT(hist_to_state_iter != hist_to_state.end());
    this_state = hist_to_state_iter->second;
    hist_to_state_iter = hist_to_state.find(next_hist);
    KALDI_ASSERT(hist_to_state_iter != hist_to_state.end());
    next_state = hist_to_state_iter->second;
    // do (*transitions)[std::pair(this_state, phone)] = next_state.
    std::pair<const std::pair<int32, int32>, int32> entry(
        std::pair<int32,int32>(this_state, phone), next_state);
    std::pair<PairMapType::iterator, bool> ret = transitions->insert(entry);
    // make sure that either it was inserted, or was already there but
    // the transition goes to the same place.
    KALDI_ASSERT(ret.second == true || *(ret.first) == entry);
  }
  std::vector<int32> zeros(opts_.ngram_order - 1, 0);
  MapType::const_iterator hist_iter = hist_to_state.find(zeros);
  KALDI_ASSERT(hist_iter != hist_to_state.end());
  return hist_iter->second;
}

void LanguageModelEstimator::OutputToFst(
    int32 initial_state,
    const LanguageModelEstimator::PairMapType &num_counts,
    const std::vector<int32> &den_counts,
    const LanguageModelEstimator::PairMapType &transitions,
    fst::StdVectorFst *fst) const {
  int32 num_states = den_counts.size();
  fst->DeleteStates();
  for (int32 i = 0; i < num_states; i++)
    fst->AddState();
  fst->SetStart(initial_state);

  int64 tot_den = std::accumulate(den_counts.begin(),
                                  den_counts.end(), 0),
      tot_num = 0;  // for self-testing code.
  double tot_logprob = 0.0;

  PairMapType::const_iterator
      iter = num_counts.begin(), end = num_counts.end();
  for (; iter != end; ++iter) {
    int32 this_state = iter->first.first,
        phone = iter->first.second,
        num_count = iter->second;
    tot_num += num_count;
    int32 den_count = den_counts[this_state];
    KALDI_ASSERT(den_count >= num_count);
    BaseFloat prob = num_count / static_cast<BaseFloat>(den_count);
    tot_logprob += num_count * log(prob);
    if (phone > 0) {
      // it's a real phone.  find out where the transition is to.
      PairMapType::const_iterator
          transitions_iter = transitions.find(iter->first);
      KALDI_ASSERT(transitions_iter != transitions.end());
      int32 dest_state = transitions_iter->second;
      fst::StdArc arc(phone, phone, fst::TropicalWeight(-log(prob)),
                      dest_state);
      fst->AddArc(this_state, arc);
    } else {
      // it's a final-prob.
      fst->SetFinal(this_state, fst::TropicalWeight(-log(prob)));
    }
  }
  KALDI_ASSERT(tot_num == tot_den);
  KALDI_LOG << "Total number of phone instances seen was " << tot_num;
  BaseFloat perplexity = exp(-(tot_logprob / tot_num));
  KALDI_LOG << "Perplexity on training data is: " << perplexity;
  KALDI_LOG << "Note: perplexity on unseen data will be infinity as there is "
            << "no smoothing.  This is by design, to reduce the number of arcs.";
  fst::Connect(fst);
  // Make sure that Connect does not delete any states.
  KALDI_ASSERT(fst->NumStates() == num_states);
  // arc-sort.  ilabel or olabel doesn't matter, it's an acceptor.
  fst::ArcSort(fst, fst::ILabelCompare<fst::StdArc>());
  KALDI_LOG << "Created phone language model with " << num_states << " states.";
}

}  // namespace chain
}  // namespace kaldi


