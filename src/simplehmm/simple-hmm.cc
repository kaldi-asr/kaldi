// hmm/simple-hmm.cc

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
//                      Johns Hopkins University (author: Guoguo Chen)
//                2016  Vimal Manohar (Johns Hopkins University)

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

#include <vector>
#include "simplehmm/simple-hmm.h"

namespace kaldi {
namespace simple_hmm {

void SimpleHmm::Initialize() {
  KALDI_ASSERT(topo_.GetPhones().size() == 1);

  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(1);
  for (int32 j = 0; j < static_cast<int32>(entry.size()); j++) {  // for each state...
    int32 pdf_class = entry[j].forward_pdf_class;
    if (pdf_class != kNoPdf) {
      states_.push_back(j);
    }
  }

  // now states_ is populated with all possible pairs
  // (hmm_state, pdf_class).
  // sort to enable reverse lookup.
  std::sort(states_.begin(), states_.end());
  // this sorting defines the transition-ids.
}

void SimpleHmm::ComputeDerived() {
  state2id_.resize(states_.size()+2);  // indexed by transition-state, which
  // is one based, but also an entry for one past end of list.

  int32 cur_transition_id = 1;
  num_pdfs_ = 0;
  for (int32 tstate = 1;
      tstate <= static_cast<int32>(states_.size()+1);  // not a typo.
      tstate++) {
    state2id_[tstate] = cur_transition_id;
    if (static_cast<size_t>(tstate) <= states_.size()) {
      int32 hmm_state = states_[tstate-1];
      const HmmTopology::HmmState &state = topo_.TopologyForPhone(1)[hmm_state];
      int32 pdf_class = state.forward_pdf_class;
      num_pdfs_ = std::max(num_pdfs_, pdf_class + 1);
      int32 my_num_ids = static_cast<int32>(state.transitions.size());
      cur_transition_id += my_num_ids;  // # trans out of this state.
    }
  }

  id2state_.resize(cur_transition_id);   // cur_transition_id is #transition-ids+1.
  for (int32 tstate = 1;
       tstate <= static_cast<int32>(states_.size()); tstate++) {
    for (int32 tid = state2id_[tstate]; tid < state2id_[tstate+1]; tid++) {
      id2state_[tid] = tstate;
    }
  }
}

void SimpleHmm::InitializeProbs() {
  log_probs_.Resize(NumTransitionIds()+1);  // one-based array, zeroth element empty.
  for (int32 trans_id = 1; trans_id <= NumTransitionIds(); trans_id++) {
    int32 trans_state = id2state_[trans_id];
    int32 trans_index = trans_id - state2id_[trans_state];
    int32 hmm_state = states_[trans_state-1];
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(1);
    KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
    BaseFloat prob = entry[hmm_state].transitions[trans_index].second;
    if (prob <= 0.0)
      KALDI_ERR << "SimpleHmm::InitializeProbs, zero "
          "probability [should remove that entry in the topology]";
    if (prob > 1.0)
      KALDI_WARN << "SimpleHmm::InitializeProbs, prob greater than one.";
    log_probs_(trans_id) = Log(prob);
  }
  ComputeDerivedOfProbs();
}

void SimpleHmm::Check() const {
  KALDI_ASSERT(topo_.GetPhones().size() == 1);

  KALDI_ASSERT(NumTransitionIds() != 0 && NumTransitionStates() != 0);
  {
    int32 sum = 0;
    for (int32 ts = 1; ts <= NumTransitionStates(); ts++) sum += NumTransitionIndices(ts);
    KALDI_ASSERT(sum == NumTransitionIds());
  }
  for (int32 tid = 1; tid <= NumTransitionIds(); tid++) {
    int32 tstate = TransitionIdToTransitionState(tid),
        index = TransitionIdToTransitionIndex(tid);
    KALDI_ASSERT(tstate > 0 && tstate <=NumTransitionStates() && index >= 0);
    KALDI_ASSERT(tid == PairToTransitionId(tstate, index));
    int32 hmm_state = TransitionStateToHmmState(tstate);
    KALDI_ASSERT(tstate == HmmStateToTransitionState(hmm_state));
    KALDI_ASSERT(log_probs_(tid) <= 0.0 &&
                 log_probs_(tid) - log_probs_(tid) == 0.0);
    // checking finite and non-positive (and not out-of-bounds).
  }

  KALDI_ASSERT(num_pdfs_ == topo_.NumPdfClasses(1));
}

SimpleHmm::SimpleHmm(
    const HmmTopology &hmm_topo): topo_(hmm_topo) {
  Initialize();
  ComputeDerived();
  InitializeProbs();
  Check();
}

int32 SimpleHmm::HmmStateToTransitionState(int32 hmm_state) const {
  // Note: if this ever gets too expensive, which is unlikely, we can refactor
  // this code to sort first on pdf_class, and then index on pdf_class, so those
  // that have the same pdf_class are in a contiguous range.
  std::vector<int32>::const_iterator iter =
      std::lower_bound(states_.begin(), states_.end(), hmm_state);
  if (iter == states_.end() || !(*iter == hmm_state)) {
    KALDI_ERR << "SimpleHmm::HmmStateToTransitionState; "
              << "HmmState " << hmm_state << " not found."
              << " (incompatible model?)";
  }
  // states_is indexed by transition_state-1, so add one.
  return static_cast<int32>((iter - states_.begin())) + 1;
}


int32 SimpleHmm::NumTransitionIndices(int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= states_.size());
  return static_cast<int32>(state2id_[trans_state+1]-state2id_[trans_state]);
}

int32 SimpleHmm::TransitionIdToTransitionState(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0 &&
               static_cast<size_t>(trans_id) < id2state_.size());
  return id2state_[trans_id];
}

int32 SimpleHmm::TransitionIdToTransitionIndex(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0 &&
               static_cast<size_t>(trans_id) < id2state_.size());
  return trans_id - state2id_[id2state_[trans_id]];
}

int32 SimpleHmm::TransitionStateToPdfClass(int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= states_.size());
  int32 hmm_state = states_[trans_state-1];
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(1);
  KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
  return entry[hmm_state].forward_pdf_class;
}

int32 SimpleHmm::TransitionStateToHmmState(int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= states_.size());
  return states_[trans_state-1];
}

int32 SimpleHmm::PairToTransitionId(int32 trans_state,
                                    int32 trans_index) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= states_.size());
  KALDI_ASSERT(trans_index < state2id_[trans_state+1] - state2id_[trans_state]);
  return state2id_[trans_state] + trans_index;
}

bool SimpleHmm::IsFinal(int32 trans_id) const {
  KALDI_ASSERT(static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];
  int32 trans_index = trans_id - state2id_[trans_state];
  int32 hmm_state = states_[trans_state-1];
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(1);
  KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
  KALDI_ASSERT(static_cast<size_t>(trans_index) <
               entry[hmm_state].transitions.size());
  // return true if the transition goes to the final state of the
  // topology entry.
  return (entry[hmm_state].transitions[trans_index].first + 1 ==
          static_cast<int32>(entry.size()));
}

// returns the self-loop transition-id,
// or zero if does not exist.
int32 SimpleHmm::SelfLoopOf(int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state-1) < states_.size());
  int32 hmm_state = states_[trans_state-1];
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(1);
  KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
  for (int32 trans_index = 0;
      trans_index < static_cast<int32>(entry[hmm_state].transitions.size());
      trans_index++)
    if (entry[hmm_state].transitions[trans_index].first == hmm_state)
      return PairToTransitionId(trans_state, trans_index);
  return 0;  // invalid transition id.
}

void SimpleHmm::ComputeDerivedOfProbs() {
  // this array indexed by transition-state with nothing in zeroth element.
  non_self_loop_log_probs_.Resize(NumTransitionStates()+1);
  for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
    int32 tid = SelfLoopOf(tstate);
    if (tid == 0) {  // no self-loop
      non_self_loop_log_probs_(tstate) = 0.0;  // log(1.0)
    } else {
      BaseFloat self_loop_prob = Exp(GetTransitionLogProb(tid)),
          non_self_loop_prob = 1.0 - self_loop_prob;
      if (non_self_loop_prob <= 0.0) {
        KALDI_WARN << "ComputeDerivedOfProbs(): non-self-loop prob is " << non_self_loop_prob;
        non_self_loop_prob = 1.0e-10;  // just so we can continue...
      }
      non_self_loop_log_probs_(tstate) = Log(non_self_loop_prob);  // will be negative.
    }
  }
}

void SimpleHmm::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<SimpleHmm>");
  topo_.Read(is, binary);
  Initialize();
  ComputeDerived();
  ExpectToken(is, binary, "<LogProbs>");
  log_probs_.Read(is, binary);
  ExpectToken(is, binary, "</LogProbs>");
  ExpectToken(is, binary, "</SimpleHmm>");
  ComputeDerivedOfProbs();
  Check();
}

void SimpleHmm::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SimpleHmm>");
  if (!binary) os << "\n";
  topo_.Write(os, binary);
  if (!binary) os << "\n";
  WriteToken(os, binary, "<LogProbs>");
  if (!binary) os << "\n";
  log_probs_.Write(os, binary);
  WriteToken(os, binary, "</LogProbs>");
  if (!binary) os << "\n";
  WriteToken(os, binary, "</SimpleHmm>");
  if (!binary) os << "\n";
}

BaseFloat SimpleHmm::GetTransitionProb(int32 trans_id) const {
  return Exp(log_probs_(trans_id));
}

BaseFloat SimpleHmm::GetTransitionLogProb(int32 trans_id) const {
  return log_probs_(trans_id);
}

BaseFloat SimpleHmm::GetNonSelfLoopLogProb(int32 trans_state) const {
  KALDI_ASSERT(trans_state != 0);
  return non_self_loop_log_probs_(trans_state);
}

BaseFloat SimpleHmm::GetTransitionLogProbIgnoringSelfLoops(
    int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0);
  KALDI_PARANOID_ASSERT(!IsSelfLoop(trans_id));
  return log_probs_(trans_id) - GetNonSelfLoopLogProb(TransitionIdToTransitionState(trans_id));
}

// stats are counts/weights, indexed by transition-id.
void SimpleHmm::MleUpdate(const Vector<double> &stats,
                          const MleSimpleHmmUpdateConfig &cfg,
                          BaseFloat *objf_impr_out,
                          BaseFloat *count_out) {
  BaseFloat count_sum = 0.0, objf_impr_sum = 0.0;
  int32 num_skipped = 0, num_floored = 0;
  KALDI_ASSERT(stats.Dim() == NumTransitionIds()+1);
  for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
    int32 n = NumTransitionIndices(tstate);
    KALDI_ASSERT(n>=1);
    if (n > 1) {  // no point updating if only one transition...
      Vector<double> counts(n);
      for (int32 tidx = 0; tidx < n; tidx++) {
        int32 tid = PairToTransitionId(tstate, tidx);
        counts(tidx) = stats(tid);
      }
      double tstate_tot = counts.Sum();
      count_sum += tstate_tot;
      if (tstate_tot < cfg.mincount) { num_skipped++; }
      else {
        Vector<BaseFloat> old_probs(n), new_probs(n);
        for (int32 tidx = 0; tidx < n; tidx++) {
          int32 tid = PairToTransitionId(tstate, tidx);
          old_probs(tidx) = new_probs(tidx) = GetTransitionProb(tid);
        }
        for (int32 tidx = 0; tidx < n; tidx++)
          new_probs(tidx) = counts(tidx) / tstate_tot;
        for (int32 i = 0; i < 3; i++) {  // keep flooring+renormalizing for 3 times..
          new_probs.Scale(1.0 / new_probs.Sum());
          for (int32 tidx = 0; tidx < n; tidx++)
            new_probs(tidx) = std::max(new_probs(tidx), cfg.floor);
        }
        // Compute objf change
        for (int32 tidx = 0; tidx < n; tidx++) {
          if (new_probs(tidx) == cfg.floor) num_floored++;
          double objf_change = counts(tidx) * (Log(new_probs(tidx))
                                               - Log(old_probs(tidx)));
          objf_impr_sum += objf_change;
        }
        // Commit updated values.
        for (int32 tidx = 0; tidx < n; tidx++) {
          int32 tid = PairToTransitionId(tstate, tidx);
          log_probs_(tid) = Log(new_probs(tidx));
          if (log_probs_(tid) - log_probs_(tid) != 0.0)
            KALDI_ERR << "Log probs is inf or NaN: error in update or bad stats?";
        }
      }
    }
  }
  KALDI_LOG << "SimpleHmm::Update, objf change is "
            << (objf_impr_sum / count_sum) << " per frame over " << count_sum
            << " frames. ";
  KALDI_LOG <<  num_floored << " probabilities floored, " << num_skipped
            << " out of " << NumTransitionStates() << " transition-states "
      "skipped due to insuffient data (it is normal to have some skipped.)";
  if (objf_impr_out) *objf_impr_out = objf_impr_sum;
  if (count_out) *count_out = count_sum;
  ComputeDerivedOfProbs();
}


// stats are counts/weights, indexed by transition-id.
void SimpleHmm::MapUpdate(const Vector<double> &stats,
                          const MapSimpleHmmUpdateConfig &cfg,
                          BaseFloat *objf_impr_out,
                          BaseFloat *count_out) {
  KALDI_ASSERT(cfg.tau > 0.0);
  BaseFloat count_sum = 0.0, objf_impr_sum = 0.0;
  KALDI_ASSERT(stats.Dim() == NumTransitionIds()+1);
  for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
    int32 n = NumTransitionIndices(tstate);
    KALDI_ASSERT(n>=1);
    if (n > 1) {  // no point updating if only one transition...
      Vector<double> counts(n);
      for (int32 tidx = 0; tidx < n; tidx++) {
        int32 tid = PairToTransitionId(tstate, tidx);
        counts(tidx) = stats(tid);
      }
      double tstate_tot = counts.Sum();
      count_sum += tstate_tot;
      Vector<BaseFloat> old_probs(n), new_probs(n);
      for (int32 tidx = 0; tidx < n; tidx++) {
        int32 tid = PairToTransitionId(tstate, tidx);
        old_probs(tidx) = new_probs(tidx) = GetTransitionProb(tid);
      }
      for (int32 tidx = 0; tidx < n; tidx++)
        new_probs(tidx) = (counts(tidx) + cfg.tau * old_probs(tidx)) /
            (cfg.tau + tstate_tot);
      // Compute objf change
      for (int32 tidx = 0; tidx < n; tidx++) {
        double objf_change = counts(tidx) * (Log(new_probs(tidx))
                                             - Log(old_probs(tidx)));
        objf_impr_sum += objf_change;
      }
      // Commit updated values.
      for (int32 tidx = 0; tidx < n; tidx++) {
        int32 tid = PairToTransitionId(tstate, tidx);
        log_probs_(tid) = Log(new_probs(tidx));
        if (log_probs_(tid) - log_probs_(tid) != 0.0)
          KALDI_ERR << "Log probs is inf or NaN: error in update or bad stats?";
      }
    }
  }
  KALDI_LOG << "Objf change is " << (objf_impr_sum / count_sum)
            << " per frame over " << count_sum
            << " frames.";
  if (objf_impr_out) *objf_impr_out = objf_impr_sum;
  if (count_out) *count_out = count_sum;
  ComputeDerivedOfProbs();
}


int32 SimpleHmm::TransitionIdToPdfClass(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0 &&
               static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];

  int32 hmm_state =  states_[trans_state-1];
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(1);
  KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
  return entry[hmm_state].forward_pdf_class;
}

int32 SimpleHmm::TransitionIdToHmmState(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0 &&
               static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];
  return states_[trans_state-1];
}

void SimpleHmm::Print(std::ostream &os,
                      const Vector<double> *occs) {
  if (occs != NULL)
    KALDI_ASSERT(occs->Dim() == NumPdfs());
  for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
    int32 hmm_state = TransitionStateToHmmState(tstate);
    int32 pdf_class = TransitionStateToPdfClass(tstate);

    os << " hmm-state = " << hmm_state;
    os << " pdf-class = " << pdf_class << '\n';
    for (int32 tidx = 0; tidx < NumTransitionIndices(tstate); tidx++) {
      int32 tid = PairToTransitionId(tstate, tidx);
      BaseFloat p = GetTransitionProb(tid);
      os << " Transition-id = " << tid << " p = " << p;
      if (occs) {
        os << " count of pdf-class = " << (*occs)(pdf_class);
      }
      // now describe what it's a transition to.
      if (IsSelfLoop(tid)) { 
        os << " [self-loop]\n";
      } else {
        const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(1);
        KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
        int32 next_hmm_state = entry[hmm_state].transitions[tidx].first;
        KALDI_ASSERT(next_hmm_state != hmm_state);
        os << " [" << hmm_state << " -> " << next_hmm_state << "]\n";
      }
    }
  }
}

bool SimpleHmm::Compatible(const SimpleHmm &other) const {
  return (topo_ == other.topo_ && states_ == other.states_ &&
          state2id_ == other.state2id_ && id2state_ == other.id2state_
          && NumPdfs() == other.NumPdfs());
}

bool SimpleHmm::IsSelfLoop(int32 trans_id) const {
  KALDI_ASSERT(static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];
  int32 trans_index = trans_id - state2id_[trans_state];
  int32 hmm_state = states_[trans_state-1];
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(1);
  KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
  return (static_cast<size_t>(trans_index) < entry[hmm_state].transitions.size()
          && entry[hmm_state].transitions[trans_index].first == hmm_state);
}

}  // end namespace simple_hmm
}  // end namespace kaldi

