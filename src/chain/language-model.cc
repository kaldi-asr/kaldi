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
  KALDI_ASSERT(opts_.ngram_order >= 2 && "--ngram-order must be >= 2");
  KALDI_ASSERT(opts_.ngram_order >= opts_.no_prune_ngram_order);
  int32 order = opts_.ngram_order;
  // 0 is used for left-context at the beginning of the file.. treat it as BOS.
  std::vector<int32> history(1, 0);
  std::vector<int32>::const_iterator iter = sentence.begin(),
      end = sentence.end();
  for (; iter != end; ++iter) {
    KALDI_ASSERT(*iter != 0);
    IncrementCount(history, *iter);
    history.push_back(*iter);
    if (history.size() >= order)
      history.erase(history.begin());
  }
  // Probability of end of sentence.  This will end up getting ignored later, but
  // it still makes a difference for probability-normalization reasons.
  IncrementCount(history, 0);
}

void LanguageModelEstimator::IncrementCount(const std::vector<int32> &history,
                                            int32 next_phone) {
  int32 lm_state_index = FindOrCreateLmStateIndexForHistory(history);
  if (lm_states_[lm_state_index].tot_count == 0) {
    num_active_lm_states_++;
  }
  lm_states_[lm_state_index].AddCount(next_phone, 1);
}

void LanguageModelEstimator::SetParentCounts() {
  int32 num_lm_states = lm_states_.size();
  for (int32 l = 0; l < num_lm_states; l++) {
    int32 this_count = lm_states_[l].tot_count;
    int32 l_iter = l;
    while (l_iter != -1) {
      lm_states_[l_iter].tot_count_with_parents += this_count;
      l_iter = lm_states_[l_iter].backoff_lmstate_index;
    }
  }
  for (int32 l = 0; l < num_lm_states; l++) {
    KALDI_ASSERT(lm_states_[l].tot_count_with_parents >=
                 lm_states_[l].tot_count);
  }
}

int32 LanguageModelEstimator::CheckActiveStates() const {
  int32 num_active_states = 0,
      num_lm_states = lm_states_.size(),
      num_basic_lm_states = 0;
  for (int32 l = 0; l < num_lm_states; l++) {
    if (lm_states_[l].tot_count != 0)
      num_active_states++;
    if (lm_states_[l].history.size() == opts_.no_prune_ngram_order - 1)
      num_basic_lm_states++;
  }
  KALDI_ASSERT(num_active_states == num_active_lm_states_);
  return num_basic_lm_states;
}

int32 LanguageModelEstimator::FindLmStateIndexForHistory(
    const std::vector<int32> &hist) const {
  MapType::const_iterator iter = hist_to_lmstate_index_.find(hist);
  if (iter == hist_to_lmstate_index_.end())
    return -1;
  else
    return iter->second;
}

int32 LanguageModelEstimator::FindNonzeroLmStateIndexForHistory(
    std::vector<int32> hist) const {
  while (1) {
    int32 l = FindLmStateIndexForHistory(hist);
    if (l == -1 || lm_states_[l].tot_count == 0) {
      // no such state or state has zero count.
      if (hist.empty())
        KALDI_ERR << "Error looking up LM state index for history "
                  << "(likely code bug)";
      hist.erase(hist.begin());  // back off.
    } else {
      return l;
    }
  }
}

int32 LanguageModelEstimator::FindOrCreateLmStateIndexForHistory(
    const std::vector<int32> &hist) {
  MapType::const_iterator iter = hist_to_lmstate_index_.find(hist);
  if (iter != hist_to_lmstate_index_.end())
    return iter->second;
  int32 ans = lm_states_.size();  // index of next element
  // next statement relies on default construct of LmState.
  lm_states_.resize(lm_states_.size() + 1);
  lm_states_.back().history = hist;
  hist_to_lmstate_index_[hist] = ans;
  // make sure backoff_lmstate_index is set, if needed.
  if (hist.size() >= opts_.no_prune_ngram_order) {
    // we need a backoff state to exist- create one if needed.
    std::vector<int32> backoff_hist(hist.begin() + 1,
                                    hist.end());

    int32 backoff_lm_state = FindOrCreateLmStateIndexForHistory(
        backoff_hist);
    lm_states_[ans].backoff_lmstate_index = backoff_lm_state;
    hist_to_lmstate_index_[backoff_hist] = backoff_lm_state;
  }
  return ans;
}

void LanguageModelEstimator::LmState::AddCount(int32 phone, int32 count) {
  std::map<int32, int32>::iterator iter = phone_to_count.find(phone);
  if (iter == phone_to_count.end())
    phone_to_count[phone] = count;
  else
    iter->second += count;
  tot_count += count;
}

void LanguageModelEstimator::LmState::Add(const LmState &other) {
  KALDI_ASSERT(&other != this);
  std::map<int32, int32>::const_iterator iter = other.phone_to_count.begin(),
      end = other.phone_to_count.end();
  for (; iter != end; ++iter)
    AddCount(iter->first, iter->second);
}

void LanguageModelEstimator::LmState::Clear() {
  phone_to_count.clear();
  tot_count = 0;
  tot_count_with_parents = false;
  backoff_allowed = false;
}

BaseFloat LanguageModelEstimator::LmState::LogLike() const {
  double ans = 0.0;
  int32 tot_count_check = 0;
  std::map<int32, int32>::const_iterator iter = phone_to_count.begin(),
      end = phone_to_count.end();
  for (; iter != end; ++iter) {
    int32 count = iter->second;
    tot_count_check += count;
    double prob = count * 1.0 / tot_count;
    ans += log(prob) * count;
  }
  KALDI_ASSERT(tot_count_check == tot_count);
  return ans;
}

void LanguageModelEstimator::InitializeQueue() {
  int32 num_lm_states = lm_states_.size();
  while (!queue_.empty()) queue_.pop();
  for (int32 l = 0; l < num_lm_states; l++) {
    lm_states_[l].backoff_allowed = BackoffAllowed(l);
    if (lm_states_[l].backoff_allowed) {
      BaseFloat like_change = BackoffLogLikelihoodChange(l);
      queue_.push(std::pair<BaseFloat,int32>(like_change, l));
    }
  }
}

BaseFloat LanguageModelEstimator::BackoffLogLikelihoodChange(
    int32 l) const {
  const LmState &lm_state = lm_states_.at(l);
  KALDI_ASSERT(lm_state.backoff_allowed && lm_state.backoff_lmstate_index >= 0);
  const LmState &backoff_lm_state = lm_states_.at(
      lm_state.backoff_lmstate_index);
  KALDI_ASSERT(lm_state.tot_count != 0);
  // if the backoff state has zero count, there would naturally be a zero
  // cost, but return -1e15 * (count of this lm state)... this encourages the
  // lowest-count state to be backed off first.
  if (backoff_lm_state.tot_count == 0)
    return -1.0e-15 * lm_state.tot_count;
  LmState sum_state(backoff_lm_state);
  sum_state.Add(lm_state);
  BaseFloat log_like_change =
      sum_state.LogLike() -
      lm_state.LogLike() -
      backoff_lm_state.LogLike();
  // log-like change should not be positive... give it a margin for round-off
  // error.
  KALDI_ASSERT(log_like_change < 0.1);
  if (log_like_change > 0.0)
    log_like_change = 0.0;
  return log_like_change;
}


void LanguageModelEstimator::DoBackoff() {
  int32 initial_active_states = num_active_lm_states_,
      target_num_lm_states = num_basic_lm_states_ + opts_.num_extra_lm_states;

  // create 3 intermediate targets and the final target.  Between each phase we'll
  // do InitializeQueue(), which will get us more exact values.
  int32 num_targets = 4;
  std::vector<int32> targets(num_targets);
  for (int32 t = 0; t < num_targets; t++) {
    // the targets get progressively closer to target_num_lm_states;
    targets[t] = initial_active_states +
        ((target_num_lm_states - initial_active_states) * (t + 1)) / num_targets;
  }
  KALDI_ASSERT(targets.back() == target_num_lm_states);

  for (int32 t = 0; t < num_targets; t++) {
    KALDI_VLOG(2) << "Backing off states, stage " << t;
    InitializeQueue();
    int32 this_target = targets[t];
    while (num_active_lm_states_ > this_target && !queue_.empty()) {
      BaseFloat like_change = queue_.top().first;
      int32 lm_state = queue_.top().second;
      queue_.pop();
      BaseFloat recomputed_like_change = BackoffLogLikelihoodChange(lm_state);
      if (!ApproxEqual(like_change, recomputed_like_change)) {
        // If it changed (i.e. we had a stale likelihood-change on the queue),
        // just put back the recomputed like-change on the queue and make no other
        // changes.
        KALDI_VLOG(2) << "Not backing off state, since like-change changed from "
                      << like_change << " to " << recomputed_like_change;
        queue_.push(std::pair<BaseFloat,int32>(recomputed_like_change, lm_state));
      } else {
        KALDI_VLOG(2) << "Backing off state with like-change = "
                      << recomputed_like_change;
        BackOffState(lm_state);
      }
    }
  }
  KALDI_LOG << "In LM [hard] backoff, target num states was "
            << num_basic_lm_states_ << " + --num-extra-lm-states="
            << opts_.num_extra_lm_states << " = " << target_num_lm_states
            << ", pruned from " << initial_active_states << " to "
            << num_active_lm_states_;
}

void LanguageModelEstimator::BackOffState(int32 l) {
  LmState &lm_state = lm_states_.at(l);
  KALDI_ASSERT(lm_state.backoff_allowed);
  KALDI_ASSERT(lm_state.backoff_lmstate_index >= 0);
  KALDI_ASSERT(lm_state.tot_count > 0);  // or shouldn't be backing it off.
  LmState &backoff_lm_state = lm_states_.at(lm_state.backoff_lmstate_index);
  bool backoff_state_had_backoff_allowed = backoff_lm_state.backoff_allowed;
  if (backoff_lm_state.tot_count != 0)
    num_active_lm_states_--;
  // add the counts of lm_state to backoff_lm_state.
  backoff_lm_state.Add(lm_state);
  // zero the counts in this lm_state.
  lm_state.Clear();
  backoff_lm_state.backoff_allowed = BackoffAllowed(
      lm_state.backoff_lmstate_index);

  if (!backoff_state_had_backoff_allowed &&
      backoff_lm_state.backoff_allowed) {
    // the backoff state would not have been in the queue, but is now allowed in
    // the queue.
    BaseFloat backoff_like_change = BackoffLogLikelihoodChange(
        lm_state.backoff_lmstate_index);
    queue_.push(std::pair<BaseFloat,int32>(backoff_like_change,
                                           lm_state.backoff_lmstate_index));
  }
}

int32 LanguageModelEstimator::AssignFstStates() {
  CheckActiveStates();
  int32 num_lm_states = lm_states_.size();
  int32 current_fst_state = 0;
  for (int32 l = 0; l < num_lm_states; l++)
    if (lm_states_[l].tot_count != 0)
      lm_states_[l].fst_state = current_fst_state++;
  KALDI_ASSERT(current_fst_state == num_active_lm_states_);
  return current_fst_state;
}

void LanguageModelEstimator::Estimate(fst::StdVectorFst *fst) {
  KALDI_LOG << "Estimating language model with --no-prune-ngram-order="
            << opts_.no_prune_ngram_order << ", --ngram-order="
            << opts_.ngram_order << ", --num-extra-lm-state="
            << opts_.num_extra_lm_states;
  SetParentCounts();
  num_basic_lm_states_ = CheckActiveStates();
  DoBackoff();
  int32 num_fst_states = AssignFstStates();
  OutputToFst(num_fst_states, fst);
}

int32 LanguageModelEstimator::FindInitialFstState() const {
  std::vector<int32> history(1, 0);
  int32 l = FindNonzeroLmStateIndexForHistory(history);
  KALDI_ASSERT(l != -1 && lm_states_[l].fst_state != -1);
  return lm_states_[l].fst_state;
}


bool LanguageModelEstimator::BackoffAllowed(int32 l) const {
  const LmState &lm_state = lm_states_.at(l);
  if (lm_state.history.size() < opts_.no_prune_ngram_order)
    return false;
  KALDI_ASSERT(lm_state.tot_count <= lm_state.tot_count_with_parents);
  if (lm_state.tot_count != lm_state.tot_count_with_parents)
    return false;
  if (lm_state.tot_count == 0)
    return false;
  // the next if-statement is an optimization where we skip the
  // following test if we know that it must always be true.
  if (lm_state.history.size() == opts_.ngram_order - 1)
    return true;
  std::map<int32, int32>::const_iterator
      iter = lm_state.phone_to_count.begin(),
      end = lm_state.phone_to_count.end();
  for (; iter != end; ++iter) {
    int32 phone = iter->first;
    if (phone != 0) {
      std::vector<int32> next_hist(lm_state.history);
      next_hist.push_back(phone);
      int32 next_lmstate = FindLmStateIndexForHistory(next_hist);
      if (next_lmstate != -1 &&
          lm_states_[next_lmstate].tot_count_with_parents != 0) {
        // backoff is not allowed because we need all the context we have
        // in order to make this transition; we can't afford to discard
        // the leftmost phone.
        return false;
      }
    }
  }
  return true;
}

void LanguageModelEstimator::OutputToFst(
    int32 num_states,
    fst::StdVectorFst *fst) const {
  KALDI_ASSERT(num_states == num_active_lm_states_);
  fst->DeleteStates();
  for (int32 i = 0; i < num_states; i++)
    fst->AddState();
  fst->SetStart(FindInitialFstState());

  int64 tot_count = 0;
  double tot_logprob = 0.0;

  int32 num_lm_states = lm_states_.size();
  // note: not all lm-states end up being 'active'.
  for (int32 l = 0; l < num_lm_states; l++) {
    const LmState &lm_state = lm_states_[l];
    if (lm_state.fst_state == -1)
      continue;
    int32 state_count = lm_state.tot_count;
    KALDI_ASSERT(state_count != 0);
    std::map<int32, int32>::const_iterator
        iter = lm_state.phone_to_count.begin(),
        end = lm_state.phone_to_count.end();
    for (; iter != end; ++iter) {
      int32 phone = iter->first, count = iter->second;
      BaseFloat logprob = log(count * 1.0 / state_count);
      tot_count += count;
      tot_logprob += logprob * count;
      if (phone == 0) {  // Interpret as final-prob.
        fst->SetFinal(lm_state.fst_state, fst::TropicalWeight(-logprob));
      } else {  // It becomes a transition.
        std::vector<int32> next_history(lm_state.history);
        next_history.push_back(phone);
        int32 dest_lm_state = FindNonzeroLmStateIndexForHistory(next_history),
            dest_fst_state = lm_states_[dest_lm_state].fst_state;
        KALDI_ASSERT(dest_fst_state != -1);
        fst->AddArc(lm_state.fst_state,
                    fst::StdArc(phone, phone, fst::TropicalWeight(-logprob),
                                dest_fst_state));
      }
    }
  }
  BaseFloat perplexity = exp(-(tot_logprob / tot_count));
  KALDI_LOG << "Total number of phone instances seen was " << tot_count;
  KALDI_LOG << "Perplexity on training data is: " << perplexity;
  KALDI_LOG << "Note: perplexity on unseen data will be infinity as there is "
            << "no smoothing.  This is by design, to reduce the number of arcs.";
  fst::Connect(fst);
  // Make sure that Connect does not delete any states.
  int32 num_states_connected = fst->NumStates();
  KALDI_ASSERT(num_states_connected == num_states);
  // arc-sort.  ilabel or olabel doesn't matter, it's an acceptor.
  fst::ArcSort(fst, fst::ILabelCompare<fst::StdArc>());
  KALDI_LOG << "Created phone language model with " << num_states
            << " states and " << fst::NumArcs(*fst) << " arcs.";
}

}  // namespace chain
}  // namespace kaldi


