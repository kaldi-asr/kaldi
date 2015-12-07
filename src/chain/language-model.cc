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
  for (; iter != end; ++iter) {
    KALDI_ASSERT(*iter != 0);
    history.push_back(*iter);
    IncrementCount(history);
    history.erase(history.begin());
  }
  // Probability of end of sentence.  This will end up getting ignored later, but
  // it still makes a difference for probability-normalization reasons.
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



void LanguageModelEstimator::Estimate(fst::StdVectorFst *fst) const {

  SetType hist_states;
  GetHistoryStates(&hist_states);

  MapType hist_to_state;
  int32 num_history_states = GetHistoryToStateMap(hist_states, &hist_to_state);
  { SetType temp; temp.swap(hist_states); }  // this won't be needed from this
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
  num_counts->clear();
  MapType::const_iterator iter = counts_.begin(), end = counts_.end();
  for (; iter != end; ++iter) {
    std::vector<int32> hist = iter->first;  // at this point it's an ngram.
    int32 count = iter->second;
    int32 phone = hist.back();
    hist.pop_back();  // now it's a history.
    int32 hist_state = GetStateForHist(hist_to_state, hist);
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

// inline static
void LanguageModelEstimator::AddCountToMap(const std::vector<int32> &key,
                                           int32 value,
                                           MapType *map) {
  MapType::iterator iter = map->find(key);
  if (iter == map->end())
    (*map)[key] = value;
  else
    iter->second += value;
}

// inline static
void LanguageModelEstimator::CopyMapToVector(
    const MapType &map,
    std::vector<std::pair<int32, std::vector<int32> > > *vec) {
  vec->clear();
  vec->reserve(map.size());
  MapType::const_iterator iter = map.begin(), end = map.end();
  for (; iter != end; ++iter) {
    vec->push_back(std::pair<int32, std::vector<int32> >(
        iter->second, iter->first));
  }
}
// inline static
void LanguageModelEstimator::CopyMapKeysToSet(
    const MapType &map,
    SetType *set) {
  set->clear();
  // in c++11 would do: set->reserve(map.size());
  MapType::const_iterator iter = map.begin(), end = map.end();
  for (; iter != end; ++iter)
    set->insert(iter->first);
}


void LanguageModelEstimator::AugmentHistoryCountsWithBackoff(
    MapType *hist_counts) const {
  std::vector<std::pair<int32, vector<int32> > > keys_and_counts;
  CopyMapToVector(*hist_counts, &keys_and_counts);

  for (std::vector<std::pair<int32, vector<int32> > >::const_iterator
           iter = keys_and_counts.begin(), end = keys_and_counts.end();
       iter != end; ++iter) {
    int32 count = iter->first;
    std::vector<int32> hist = iter->second;
    while (hist.size() > 1) {
      hist.erase(hist.begin());
      AddCountToMap(hist, count, hist_counts);
    }
  }
}

void LanguageModelEstimator::GetHistoryStates(SetType *hist_set) const {
  MapType history_counts;
  GetHistoryCounts(&history_counts);
  CopyMapKeysToSet(history_counts, hist_set);
  int32 orig_num_lm_states = hist_set->size();
  AugmentHistoryCountsWithBackoff(&history_counts);
  std::vector<std::pair<int32, std::vector<int32> > > counts_vec;
  CopyMapToVector(history_counts, &counts_vec);
  std::sort(counts_vec.begin(), counts_vec.end(), HistoryCountCompare());

  int32 size = counts_vec.size();
  for (int32 i = 0; i < size && hist_set->size() > opts_.num_lm_states; i++) {
    const std::vector<int32> &hist_state = counts_vec[i].second;
    if (hist_state.size() <= 1)
      continue;  // we never prune bigram history-states.
                 // this keeps the transitions between states sparse.
    SetType::iterator iter = hist_set->find(hist_state);
    if (iter != hist_set->end()) {
      // if this history-state is actually in the set...
      std::vector<int32> backoff_state(hist_state.begin() + 1,
                                       hist_state.end());
      hist_set->erase(iter);  // erase this history-state.
      hist_set->insert(backoff_state);  // ensure the relevant backoff state
                                           // is present in 'hist_set'.
    } else {
      // we don't expect this line to be reached.
      KALDI_WARN << "History-state not found in the set (unexpected)";
    }
  }
  KALDI_LOG << "Reduced number of LM history-states from "
            << orig_num_lm_states << " to " << hist_set->size();
}

int32 LanguageModelEstimator::GetHistoryToStateMap(
    const LanguageModelEstimator::SetType &hist_set,
    LanguageModelEstimator::MapType *hist_to_state) const {

  SetType::const_iterator iter = hist_set.begin(), end = hist_set.end();
  hist_to_state->clear();
  int32 cur_state = 0;
  for (; iter != end; ++iter)
    (*hist_to_state)[*iter] = cur_state++;
  return cur_state;
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
    if (phone == 0)
      continue;
    int32 this_state = GetStateForHist(hist_to_state, this_hist),
          next_state = GetStateForHist(hist_to_state, next_hist);

    // do (*transitions)[std::pair(this_state, phone)] = next_state.
    std::pair<const std::pair<int32, int32>, int32> entry(
        std::pair<int32,int32>(this_state, phone), next_state);
    std::pair<PairMapType::iterator, bool> ret = transitions->insert(entry);
    // make sure that either it was inserted, or was already there but the
    // transition goes to the same place.  Failure could possibly mean an issue
    // where deleted history-states in a 'disallowed' way, e.g.  deleting a
    // state that was the successor-state for some other state that has not yet
    // been deleted; but the approach of sorting on counts (and if the counts
    // are the same, then deleting the longest history-state first, i.e. with
    // the most words) should prevent this from happening.
    KALDI_ASSERT(ret.second == true || *(ret.first) == entry);
  }
  std::vector<int32> zeros(opts_.ngram_order - 1, 0);
  int32 zero_state = GetStateForHist(hist_to_state, zeros);
  return zero_state;
}

// static
int32 LanguageModelEstimator::GetStateForHist(const MapType &hist_to_state,
                                              std::vector<int32> hist) {

  while (true) {
    if (hist.size() == 0)  // you'll have to figure out the sequence from the
                           // stack.
      KALDI_ERR << "Error getting state for history.  Code error in LM code.";
    MapType::const_iterator iter = hist_to_state.find(hist);
    if (iter == hist_to_state.end())
      hist.erase(hist.begin());  // back off.
    else
      return iter->second;
  }
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


