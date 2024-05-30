// hmm/transition-model.cc

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
//        Johns Hopkins University (author: Guoguo Chen)

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
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

namespace kaldi {

void TransitionModel::ComputeTuples(const ContextDependencyInterface &ctx_dep) {
  if (IsHmm())
    ComputeTuplesIsHmm(ctx_dep);
  else
    ComputeTuplesNotHmm(ctx_dep);

  // now tuples_ is populated with all possible tuples of (phone, hmm_state, pdf, self_loop_pdf).
  std::sort(tuples_.begin(), tuples_.end());  // sort to enable reverse lookup.
  // this sorting defines the transition-ids.
}

void TransitionModel::ComputeTuplesIsHmm(const ContextDependencyInterface &ctx_dep) {
  const std::vector<int32> &phones = topo_.GetPhones();
  KALDI_ASSERT(!phones.empty());

  // this is the case for normal models. but not for chain models
  std::vector<std::vector<std::pair<int32, int32> > > pdf_info;
  std::vector<int32> num_pdf_classes( 1 + *std::max_element(phones.begin(), phones.end()), -1);
  for (size_t i = 0; i < phones.size(); i++)
    num_pdf_classes[phones[i]] = topo_.NumPdfClasses(phones[i]);
  ctx_dep.GetPdfInfo(phones, num_pdf_classes, &pdf_info);
  // pdf_info is list indexed by pdf of which (phone, pdf_class) it
  // can correspond to.

  std::map<std::pair<int32, int32>, std::vector<int32> > to_hmm_state_list;
  // to_hmm_state_list is a map from (phone, pdf_class) to the list
  // of hmm-states in the HMM for that phone that that (phone, pdf-class)
  // can correspond to.
  for (size_t i = 0; i < phones.size(); i++) {  // setting up to_hmm_state_list.
    int32 phone = phones[i];
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
    for (int32 j = 0; j < static_cast<int32>(entry.size()); j++) {  // for each state...
      int32 pdf_class = entry[j].forward_pdf_class;
      if (pdf_class != kNoPdf) {
        to_hmm_state_list[std::make_pair(phone, pdf_class)].push_back(j);
      }
    }
  }

  for (int32 pdf = 0; pdf < static_cast<int32>(pdf_info.size()); pdf++) {
    for (size_t j = 0; j < pdf_info[pdf].size(); j++) {
      int32 phone = pdf_info[pdf][j].first,
            pdf_class = pdf_info[pdf][j].second;
      const std::vector<int32> &state_vec = to_hmm_state_list[std::make_pair(phone, pdf_class)];
      KALDI_ASSERT(!state_vec.empty());
      // state_vec is a list of the possible HMM-states that emit this
      // pdf_class.
      for (size_t k = 0; k < state_vec.size(); k++) {
        int32 hmm_state = state_vec[k];
        tuples_.push_back(Tuple(phone, hmm_state, pdf, pdf));
      }
    }
  }
}

void TransitionModel::ComputeTuplesNotHmm(const ContextDependencyInterface &ctx_dep) {
  const std::vector<int32> &phones = topo_.GetPhones();
  KALDI_ASSERT(!phones.empty());

  // pdf_info is a set of lists indexed by phone. Each list is indexed by
  // (pdf-class, self-loop pdf-class) of each state of that phone, and the element
  // is a list of possible (pdf, self-loop pdf) pairs that (pdf-class, self-loop pdf-class)
  // pair generates.
  std::vector<std::vector<std::vector<std::pair<int32, int32> > > > pdf_info;
  // pdf_class_pairs is a set of lists indexed by phone. Each list stores
  // (pdf-class, self-loop pdf-class) of each state of that phone.
  std::vector<std::vector<std::pair<int32, int32> > > pdf_class_pairs;
  pdf_class_pairs.resize(1 + *std::max_element(phones.begin(), phones.end()));
  for (size_t i = 0; i < phones.size(); i++) {
    int32 phone = phones[i];
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
    for (int32 j = 0; j < static_cast<int32>(entry.size()); j++) {  // for each state...
      int32 forward_pdf_class = entry[j].forward_pdf_class, self_loop_pdf_class = entry[j].self_loop_pdf_class;
      if (forward_pdf_class != kNoPdf)
        pdf_class_pairs[phone].push_back(std::make_pair(forward_pdf_class, self_loop_pdf_class));
    }
  }
  ctx_dep.GetPdfInfo(phones, pdf_class_pairs, &pdf_info);

  std::vector<std::map<std::pair<int32, int32>, std::vector<int32> > > to_hmm_state_list;
  to_hmm_state_list.resize(1 + *std::max_element(phones.begin(), phones.end()));
  // to_hmm_state_list is a phone-indexed set of maps from (pdf-class, self-loop pdf_class) to the list
  // of hmm-states in the HMM for that phone that that (pdf-class, self-loop pdf-class)
  // can correspond to.
  for (size_t i = 0; i < phones.size(); i++) {  // setting up to_hmm_state_list.
    int32 phone = phones[i];
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
    std::map<std::pair<int32, int32>, std::vector<int32> > phone_to_hmm_state_list;
    for (int32 j = 0; j < static_cast<int32>(entry.size()); j++) {  // for each state...
      int32 forward_pdf_class = entry[j].forward_pdf_class, self_loop_pdf_class = entry[j].self_loop_pdf_class;
      if (forward_pdf_class != kNoPdf) {
        phone_to_hmm_state_list[std::make_pair(forward_pdf_class, self_loop_pdf_class)].push_back(j);
      }
    }
    to_hmm_state_list[phone] = phone_to_hmm_state_list;
  }

  for (int32 i = 0; i < phones.size(); i++) {
    int32 phone = phones[i];
    for (int32 j = 0; j < static_cast<int32>(pdf_info[phone].size()); j++) {
      int32 pdf_class = pdf_class_pairs[phone][j].first,
            self_loop_pdf_class = pdf_class_pairs[phone][j].second;
      const std::vector<int32> &state_vec =
              to_hmm_state_list[phone][std::make_pair(pdf_class, self_loop_pdf_class)];
      KALDI_ASSERT(!state_vec.empty());
      for (size_t k = 0; k < state_vec.size(); k++) {
        int32 hmm_state = state_vec[k];
        for (size_t m = 0; m < pdf_info[phone][j].size(); m++) {
          int32 pdf = pdf_info[phone][j][m].first,
            self_loop_pdf = pdf_info[phone][j][m].second;
          tuples_.push_back(Tuple(phone, hmm_state, pdf, self_loop_pdf));
        }
      }
    }
  }
}

void TransitionModel::ComputeDerived() {
  state2id_.resize(tuples_.size()+2);  // indexed by transition-state, which
  // is one based, but also an entry for one past end of list.

  int32 cur_transition_id = 1;
  num_pdfs_ = 0;
  for (int32 tstate = 1;
      tstate <= static_cast<int32>(tuples_.size()+1);  // not a typo.
      tstate++) {
    state2id_[tstate] = cur_transition_id;
    if (static_cast<size_t>(tstate) <= tuples_.size()) {
      int32 phone = tuples_[tstate-1].phone,
          hmm_state = tuples_[tstate-1].hmm_state,
          forward_pdf = tuples_[tstate-1].forward_pdf,
          self_loop_pdf = tuples_[tstate-1].self_loop_pdf;
      num_pdfs_ = std::max(num_pdfs_, 1 + forward_pdf);
      num_pdfs_ = std::max(num_pdfs_, 1 + self_loop_pdf);
      const HmmTopology::HmmState &state = topo_.TopologyForPhone(phone)[hmm_state];
      int32 my_num_ids = static_cast<int32>(state.transitions.size());
      cur_transition_id += my_num_ids;  // # trans out of this state.
    }
  }

  id2state_.resize(cur_transition_id);   // cur_transition_id is #transition-ids+1.
  id2pdf_id_.resize(cur_transition_id);
  for (int32 tstate = 1; tstate <= static_cast<int32>(tuples_.size()); tstate++) {
    for (int32 tid = state2id_[tstate]; tid < state2id_[tstate+1]; tid++) {
      id2state_[tid] = tstate;
      if (IsSelfLoop(tid))
        id2pdf_id_[tid] = tuples_[tstate-1].self_loop_pdf;
      else
        id2pdf_id_[tid] = tuples_[tstate-1].forward_pdf;
    }
  }

  // The following statements put copies a large number in the region of memory
  // past the end of the id2pdf_id_ array, while leaving the array as it was
  // before.  The goal of this is to speed up decoding by disabling a check
  // inside TransitionIdToPdf() that the transition-id was within the correct
  // range.
  int32 num_big_numbers = std::min<int32>(2000, cur_transition_id);
  id2pdf_id_.resize(cur_transition_id + num_big_numbers,
                    std::numeric_limits<int32>::max());
  id2pdf_id_.resize(cur_transition_id);
}

void TransitionModel::InitializeProbs() {
  log_probs_.Resize(NumTransitionIds()+1);  // one-based array, zeroth element empty.
  for (int32 trans_id = 1; trans_id <= NumTransitionIds(); trans_id++) {
    int32 trans_state = id2state_[trans_id];
    int32 trans_index = trans_id - state2id_[trans_state];
    const Tuple &tuple = tuples_[trans_state-1];
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(tuple.phone);
    KALDI_ASSERT(static_cast<size_t>(tuple.hmm_state) < entry.size());
    BaseFloat prob = entry[tuple.hmm_state].transitions[trans_index].second;
    if (prob <= 0.0)
      KALDI_ERR << "TransitionModel::InitializeProbs, zero "
          "probability [should remove that entry in the topology]";
    if (prob > 1.0)
      KALDI_WARN << "TransitionModel::InitializeProbs, prob greater than one.";
    log_probs_(trans_id) = Log(prob);
  }
  ComputeDerivedOfProbs();
}

void TransitionModel::Check() const {
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
    int32 phone = TransitionStateToPhone(tstate),
        hmm_state = TransitionStateToHmmState(tstate),
        forward_pdf = TransitionStateToForwardPdf(tstate),
        self_loop_pdf = TransitionStateToSelfLoopPdf(tstate);
    KALDI_ASSERT(tstate == TupleToTransitionState(phone, hmm_state, forward_pdf, self_loop_pdf));
    KALDI_ASSERT(log_probs_(tid) <= 0.0 && log_probs_(tid) - log_probs_(tid) == 0.0);
    // checking finite and non-positive (and not out-of-bounds).
  }
}

bool TransitionModel::IsHmm() const {
  const std::vector<int32> &phones = topo_.GetPhones();
  KALDI_ASSERT(!phones.empty());
  for (size_t i = 0; i < phones.size(); i++) {
    int32 phone = phones[i];
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
    for (int32 j = 0; j < static_cast<int32>(entry.size()); j++) {  // for each state...
      if (entry[j].forward_pdf_class != entry[j].self_loop_pdf_class)
        return false;
    }
  }
  return true;
}

TransitionModel::TransitionModel(const ContextDependencyInterface &ctx_dep,
                                 const HmmTopology &hmm_topo): topo_(hmm_topo) {
  // First thing is to get all possible tuples.
  ComputeTuples(ctx_dep);
  ComputeDerived();
  InitializeProbs();
  Check();
}

int32 TransitionModel::TupleToTransitionState(int32 phone, int32 hmm_state, int32 pdf, int32 self_loop_pdf) const {
  Tuple tuple(phone, hmm_state, pdf, self_loop_pdf);
  // Note: if this ever gets too expensive, which is unlikely, we can refactor
  // this code to sort first on pdf, and then index on pdf, so those
  // that have the same pdf are in a contiguous range.
  std::vector<Tuple>::const_iterator iter =
      std::lower_bound(tuples_.begin(), tuples_.end(), tuple);
  if (iter == tuples_.end() || !(*iter == tuple)) {
    KALDI_ERR << "TransitionModel::TupleToTransitionState, tuple not found."
              << " (incompatible tree and model?)";
  }
  // tuples_ is indexed by transition_state-1, so add one.
  return static_cast<int32>((iter - tuples_.begin())) + 1;
}


int32 TransitionModel::NumTransitionIndices(int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  return static_cast<int32>(state2id_[trans_state+1]-state2id_[trans_state]);
}

int32 TransitionModel::TransitionIdToTransitionState(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0 &&  static_cast<size_t>(trans_id) < id2state_.size());
  return id2state_[trans_id];
}

int32 TransitionModel::TransitionIdToTransitionIndex(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_.size());
  return trans_id - state2id_[id2state_[trans_id]];
}

int32 TransitionModel::TransitionStateToPhone(int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  return tuples_[trans_state-1].phone;
}

int32 TransitionModel::TransitionStateToForwardPdf(int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  return tuples_[trans_state-1].forward_pdf;
}

int32 TransitionModel::TransitionStateToForwardPdfClass(
    int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  const Tuple &t = tuples_[trans_state-1];
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(t.phone);
  KALDI_ASSERT(static_cast<size_t>(t.hmm_state) < entry.size());
  return entry[t.hmm_state].forward_pdf_class;
}


int32 TransitionModel::TransitionStateToSelfLoopPdfClass(
    int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  const Tuple &t = tuples_[trans_state-1];
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(t.phone);
  KALDI_ASSERT(static_cast<size_t>(t.hmm_state) < entry.size());
  return entry[t.hmm_state].self_loop_pdf_class;
}


int32 TransitionModel::TransitionStateToSelfLoopPdf(int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  return tuples_[trans_state-1].self_loop_pdf;
}

int32 TransitionModel::TransitionStateToHmmState(int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  return tuples_[trans_state-1].hmm_state;
}

int32 TransitionModel::PairToTransitionId(int32 trans_state, int32 trans_index) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  KALDI_ASSERT(trans_index < state2id_[trans_state+1] - state2id_[trans_state]);
  return state2id_[trans_state] + trans_index;
}

int32 TransitionModel::NumPhones() const {
  int32 num_trans_state = tuples_.size();
  int32 max_phone_id = 0;
  for (int32 i = 0; i < num_trans_state; ++i) {
    if (tuples_[i].phone > max_phone_id)
      max_phone_id = tuples_[i].phone;
  }
  return max_phone_id;
}


bool TransitionModel::IsFinal(int32 trans_id) const {
  KALDI_ASSERT(static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];
  int32 trans_index = trans_id - state2id_[trans_state];
  const Tuple &tuple = tuples_[trans_state-1];
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(tuple.phone);
  KALDI_ASSERT(static_cast<size_t>(tuple.hmm_state) < entry.size());
  KALDI_ASSERT(static_cast<size_t>(tuple.hmm_state) < entry.size());
  KALDI_ASSERT(static_cast<size_t>(trans_index) <
               entry[tuple.hmm_state].transitions.size());
  // return true if the transition goes to the final state of the
  // topology entry.
  return (entry[tuple.hmm_state].transitions[trans_index].first + 1 ==
          static_cast<int32>(entry.size()));
}



int32 TransitionModel::SelfLoopOf(int32 trans_state) const {  // returns the self-loop transition-id,
  KALDI_ASSERT(static_cast<size_t>(trans_state-1) < tuples_.size());
  const Tuple &tuple = tuples_[trans_state-1];
  // or zero if does not exist.
  int32 phone = tuple.phone, hmm_state = tuple.hmm_state;
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
  KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
  for (int32 trans_index = 0;
      trans_index < static_cast<int32>(entry[hmm_state].transitions.size());
      trans_index++)
    if (entry[hmm_state].transitions[trans_index].first == hmm_state)
      return PairToTransitionId(trans_state, trans_index);
  return 0;  // invalid transition id.
}

void TransitionModel::ComputeDerivedOfProbs() {
  non_self_loop_log_probs_.Resize(NumTransitionStates()+1);  // this array indexed
  //  by transition-state with nothing in zeroth element.
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

void TransitionModel::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<TransitionModel>");
  topo_.Read(is, binary);
  std::string token;
  ReadToken(is, binary, &token);
  int32 size;
  ReadBasicType(is, binary, &size);
  tuples_.resize(size);
  for (int32 i = 0; i < size; i++) {
    ReadBasicType(is, binary, &(tuples_[i].phone));
    ReadBasicType(is, binary, &(tuples_[i].hmm_state));
    ReadBasicType(is, binary, &(tuples_[i].forward_pdf));
    if (token == "<Tuples>")
      ReadBasicType(is, binary, &(tuples_[i].self_loop_pdf));
    else if (token == "<Triples>")
      tuples_[i].self_loop_pdf = tuples_[i].forward_pdf;
  }
  ReadToken(is, binary, &token);
  KALDI_ASSERT(token == "</Triples>" || token == "</Tuples>");
  ComputeDerived();
  ExpectToken(is, binary, "<LogProbs>");
  log_probs_.Read(is, binary);
  ExpectToken(is, binary, "</LogProbs>");
  ExpectToken(is, binary, "</TransitionModel>");
  ComputeDerivedOfProbs();
  Check();
}

void TransitionModel::Write(std::ostream &os, bool binary) const {
  bool is_hmm = IsHmm();
  WriteToken(os, binary, "<TransitionModel>");
  if (!binary) os << "\n";
  topo_.Write(os, binary);
  if (is_hmm)
    WriteToken(os, binary, "<Triples>");
  else
    WriteToken(os, binary, "<Tuples>");
  WriteBasicType(os, binary, static_cast<int32>(tuples_.size()));
  if (!binary) os << "\n";
  for (int32 i = 0; i < static_cast<int32> (tuples_.size()); i++) {
    WriteBasicType(os, binary, tuples_[i].phone);
    WriteBasicType(os, binary, tuples_[i].hmm_state);
    WriteBasicType(os, binary, tuples_[i].forward_pdf);
    if (!is_hmm)
      WriteBasicType(os, binary, tuples_[i].self_loop_pdf);
    if (!binary) os << "\n";
  }
  if (is_hmm)
    WriteToken(os, binary, "</Triples>");
  else
    WriteToken(os, binary, "</Tuples>");
  if (!binary) os << "\n";
  WriteToken(os, binary, "<LogProbs>");
  if (!binary) os << "\n";
  log_probs_.Write(os, binary);
  WriteToken(os, binary, "</LogProbs>");
  if (!binary) os << "\n";
  WriteToken(os, binary, "</TransitionModel>");
  if (!binary) os << "\n";
}

BaseFloat TransitionModel::GetTransitionProb(int32 trans_id) const {
  return Exp(log_probs_(trans_id));
}

BaseFloat TransitionModel::GetTransitionLogProb(int32 trans_id) const {
  return log_probs_(trans_id);
}

BaseFloat TransitionModel::GetNonSelfLoopLogProb(int32 trans_state) const {
  KALDI_ASSERT(trans_state != 0);
  return non_self_loop_log_probs_(trans_state);
}

BaseFloat TransitionModel::GetTransitionLogProbIgnoringSelfLoops(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0);
  KALDI_PARANOID_ASSERT(!IsSelfLoop(trans_id));
  return log_probs_(trans_id) - GetNonSelfLoopLogProb(TransitionIdToTransitionState(trans_id));
}

// stats are counts/weights, indexed by transition-id.
void TransitionModel::MleUpdate(const Vector<double> &stats,
                                const MleTransitionUpdateConfig &cfg,
                                BaseFloat *objf_impr_out,
                                BaseFloat *count_out) {
  if (cfg.share_for_pdfs) {
    MleUpdateShared(stats, cfg, objf_impr_out, count_out);
    return;
  }
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
  KALDI_LOG << "TransitionModel::Update, objf change is "
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
void TransitionModel::MapUpdate(const Vector<double> &stats,
                                const MapTransitionUpdateConfig &cfg,
                                BaseFloat *objf_impr_out,
                                BaseFloat *count_out) {
  KALDI_ASSERT(cfg.tau > 0.0);
  if (cfg.share_for_pdfs) {
    MapUpdateShared(stats, cfg, objf_impr_out, count_out);
    return;
  }
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



/// This version of the Update() function is for if the user specifies
/// --share-for-pdfs=true.  We share the transitions for all states that
/// share the same pdf.
void TransitionModel::MleUpdateShared(const Vector<double> &stats,
                                      const MleTransitionUpdateConfig &cfg,
                                      BaseFloat *objf_impr_out,
                                      BaseFloat *count_out) {
  KALDI_ASSERT(cfg.share_for_pdfs);

  BaseFloat count_sum = 0.0, objf_impr_sum = 0.0;
  int32 num_skipped = 0, num_floored = 0;
  KALDI_ASSERT(stats.Dim() == NumTransitionIds()+1);
  std::map<int32, std::set<int32> > pdf_to_tstate;

  for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
    int32 pdf = TransitionStateToForwardPdf(tstate);
    pdf_to_tstate[pdf].insert(tstate);
    if (!IsHmm()) {
      pdf = TransitionStateToSelfLoopPdf(tstate);
      pdf_to_tstate[pdf].insert(tstate);
    }
  }
  std::map<int32, std::set<int32> >::iterator map_iter;
  for (map_iter = pdf_to_tstate.begin();
       map_iter != pdf_to_tstate.end();
       ++map_iter) {
    // map_iter->first is pdf-id... not needed.
    const std::set<int32> &tstates = map_iter->second;
    KALDI_ASSERT(!tstates.empty());
    int32 one_tstate = *(tstates.begin());
    int32 n = NumTransitionIndices(one_tstate);
    KALDI_ASSERT(n >= 1);
    if (n > 1) { // Only update if >1 transition...
      Vector<double> counts(n);
      for (std::set<int32>::const_iterator iter = tstates.begin();
           iter != tstates.end();
           ++iter) {
        int32 tstate = *iter;
        if (NumTransitionIndices(tstate) != n)
          KALDI_ERR << "Mismatch in #transition indices: you cannot "
              "use the --share-for-pdfs option with this topology "
              "and sharing scheme.";
        for (int32 tidx = 0; tidx < n; tidx++) {
          int32 tid = PairToTransitionId(tstate, tidx);
          counts(tidx) += stats(tid);
        }
      }
      double pdf_tot = counts.Sum();
      count_sum += pdf_tot;
      if (pdf_tot < cfg.mincount) { num_skipped++; }
      else {
        // Note: when calculating objf improvement, we
        // assume we previously had the same tying scheme so
        // we can get the params from one_tstate and they're valid
        // for all.
        Vector<BaseFloat> old_probs(n), new_probs(n);
        for (int32 tidx = 0; tidx < n; tidx++) {
          int32 tid = PairToTransitionId(one_tstate, tidx);
          old_probs(tidx) = new_probs(tidx) = GetTransitionProb(tid);
        }
        for (int32 tidx = 0; tidx < n; tidx++)
          new_probs(tidx) = counts(tidx) / pdf_tot;
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
        for (std::set<int32>::const_iterator iter = tstates.begin();
             iter != tstates.end();
             ++iter) {
          int32 tstate = *iter;
          for (int32 tidx = 0; tidx < n; tidx++) {
            int32 tid = PairToTransitionId(tstate, tidx);
            log_probs_(tid) = Log(new_probs(tidx));
            if (log_probs_(tid) - log_probs_(tid) != 0.0)
              KALDI_ERR << "Log probs is inf or NaN: error in update or bad stats?";
          }
        }
      }
    }
  }
  KALDI_LOG << "Objf change is " << (objf_impr_sum / count_sum)
            << " per frame over " << count_sum << " frames; "
            << num_floored << " probabilities floored, "
            << num_skipped << " pdf-ids skipped due to insuffient data.";
  if (objf_impr_out) *objf_impr_out = objf_impr_sum;
  if (count_out) *count_out = count_sum;
  ComputeDerivedOfProbs();
}


/// This version of the MapUpdate() function is for if the user specifies
/// --share-for-pdfs=true.  We share the transitions for all states that
/// share the same pdf.
void TransitionModel::MapUpdateShared(const Vector<double> &stats,
                                      const MapTransitionUpdateConfig &cfg,
                                      BaseFloat *objf_impr_out,
                                      BaseFloat *count_out) {
  KALDI_ASSERT(cfg.share_for_pdfs);

  BaseFloat count_sum = 0.0, objf_impr_sum = 0.0;
  KALDI_ASSERT(stats.Dim() == NumTransitionIds()+1);
  std::map<int32, std::set<int32> > pdf_to_tstate;

  for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
    int32 pdf = TransitionStateToForwardPdf(tstate);
    pdf_to_tstate[pdf].insert(tstate);
    if (!IsHmm()) {
      pdf = TransitionStateToSelfLoopPdf(tstate);
      pdf_to_tstate[pdf].insert(tstate);
    }
  }
  std::map<int32, std::set<int32> >::iterator map_iter;
  for (map_iter = pdf_to_tstate.begin();
       map_iter != pdf_to_tstate.end();
       ++map_iter) {
    // map_iter->first is pdf-id... not needed.
    const std::set<int32> &tstates = map_iter->second;
    KALDI_ASSERT(!tstates.empty());
    int32 one_tstate = *(tstates.begin());
    int32 n = NumTransitionIndices(one_tstate);
    KALDI_ASSERT(n >= 1);
    if (n > 1) { // Only update if >1 transition...
      Vector<double> counts(n);
      for (std::set<int32>::const_iterator iter = tstates.begin();
           iter != tstates.end();
           ++iter) {
        int32 tstate = *iter;
        if (NumTransitionIndices(tstate) != n)
          KALDI_ERR << "Mismatch in #transition indices: you cannot "
              "use the --share-for-pdfs option with this topology "
              "and sharing scheme.";
        for (int32 tidx = 0; tidx < n; tidx++) {
          int32 tid = PairToTransitionId(tstate, tidx);
          counts(tidx) += stats(tid);
        }
      }
      double pdf_tot = counts.Sum();
      count_sum += pdf_tot;

      // Note: when calculating objf improvement, we
      // assume we previously had the same tying scheme so
      // we can get the params from one_tstate and they're valid
      // for all.
      Vector<BaseFloat> old_probs(n), new_probs(n);
      for (int32 tidx = 0; tidx < n; tidx++) {
        int32 tid = PairToTransitionId(one_tstate, tidx);
        old_probs(tidx) = new_probs(tidx) = GetTransitionProb(tid);
      }
      for (int32 tidx = 0; tidx < n; tidx++)
        new_probs(tidx) = (counts(tidx) + old_probs(tidx) * cfg.tau) /
            (pdf_tot + cfg.tau);
      // Compute objf change
      for (int32 tidx = 0; tidx < n; tidx++) {
        double objf_change = counts(tidx) * (Log(new_probs(tidx))
                                             - Log(old_probs(tidx)));
        objf_impr_sum += objf_change;
      }
      // Commit updated values.
      for (std::set<int32>::const_iterator iter = tstates.begin();
           iter != tstates.end();
           ++iter) {
        int32 tstate = *iter;
        for (int32 tidx = 0; tidx < n; tidx++) {
          int32 tid = PairToTransitionId(tstate, tidx);
          log_probs_(tid) = Log(new_probs(tidx));
          if (log_probs_(tid) - log_probs_(tid) != 0.0)
            KALDI_ERR << "Log probs is inf or NaN: error in update or bad stats?";
        }
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

bool TransitionModel::TransitionIdsEquivalent(int32_t trans_id1,
                                              int32_t trans_id2) const {
  return TransitionIdToTransitionState(trans_id1) ==
    TransitionIdToTransitionState(trans_id2);
}

bool TransitionModel::TransitionIdIsStartOfPhone(int32_t trans_id) const {
  return TransitionIdToHmmState(trans_id) == 0;
}

const std::vector<int32>& TransitionModel::TransitionIdToPdfArray() const {
  return id2pdf_id_;
}

int32 TransitionModel::TransitionIdToPhone(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];
  return tuples_[trans_state-1].phone;
}

int32 TransitionModel::TransitionIdToPdfClass(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];

  const Tuple &t = tuples_[trans_state-1];
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(t.phone);
  KALDI_ASSERT(static_cast<size_t>(t.hmm_state) < entry.size());
  if (IsSelfLoop(trans_id))
    return entry[t.hmm_state].self_loop_pdf_class;
  else
    return entry[t.hmm_state].forward_pdf_class;
}


int32 TransitionModel::TransitionIdToHmmState(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];
  const Tuple &t = tuples_[trans_state-1];
  return t.hmm_state;
}

void TransitionModel::Print(std::ostream &os,
                            const std::vector<std::string> &phone_names,
                            const Vector<double> *occs) {
  if (occs != NULL)
    KALDI_ASSERT(occs->Dim() == NumPdfs());
  bool is_hmm = IsHmm();
  for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
    const Tuple &tuple = tuples_[tstate-1];
    KALDI_ASSERT(static_cast<size_t>(tuple.phone) < phone_names.size());
    std::string phone_name = phone_names[tuple.phone];

    os << "Transition-state " << tstate << ": phone = " << phone_name
       << " hmm-state = " << tuple.hmm_state;
    if (is_hmm)
      os << " pdf = " << tuple.forward_pdf << '\n';
    else
      os << " forward-pdf = " << tuple.forward_pdf << " self-loop-pdf = "
         << tuple.self_loop_pdf << '\n';
    for (int32 tidx = 0; tidx < NumTransitionIndices(tstate); tidx++) {
      int32 tid = PairToTransitionId(tstate, tidx);
      BaseFloat p = GetTransitionProb(tid);
      os << " Transition-id = " << tid << " p = " << p;
      if (occs != NULL) {
        if (IsSelfLoop(tid))
          os << " count of pdf = " << (*occs)(tuple.self_loop_pdf);
        else
          os << " count of pdf = " << (*occs)(tuple.forward_pdf);
      }
      // now describe what it's a transition to.
      if (IsSelfLoop(tid)) os << " [self-loop]\n";
      else {
        int32 hmm_state = tuple.hmm_state;
        const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(tuple.phone);
        KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
        int32 next_hmm_state = entry[hmm_state].transitions[tidx].first;
        KALDI_ASSERT(next_hmm_state != hmm_state);
        os << " [" << hmm_state << " -> " << next_hmm_state << "]\n";
      }
    }
  }
}

bool GetPdfsForPhones(const TransitionModel &trans_model,
                      const std::vector<int32> &phones,
                      std::vector<int32> *pdfs) {
  KALDI_ASSERT(IsSortedAndUniq(phones));
  KALDI_ASSERT(pdfs != NULL);
  pdfs->clear();
  for (int32 tstate = 1; tstate <= trans_model.NumTransitionStates(); tstate++) {
    if (std::binary_search(phones.begin(), phones.end(),
             trans_model.TransitionStateToPhone(tstate))) {
      pdfs->push_back(trans_model.TransitionStateToForwardPdf(tstate));
      pdfs->push_back(trans_model.TransitionStateToSelfLoopPdf(tstate));
    }
  }
  SortAndUniq(pdfs);

  for (int32 tstate = 1; tstate <= trans_model.NumTransitionStates(); tstate++)
    if ((std::binary_search(pdfs->begin(), pdfs->end(),
                          trans_model.TransitionStateToForwardPdf(tstate)) ||
         std::binary_search(pdfs->begin(), pdfs->end(),
                          trans_model.TransitionStateToSelfLoopPdf(tstate)))
       && !std::binary_search(phones.begin(), phones.end(),
                              trans_model.TransitionStateToPhone(tstate)))
      return false;
  return true;
}

bool GetPhonesForPdfs(const TransitionModel &trans_model,
                     const std::vector<int32> &pdfs,
                     std::vector<int32> *phones) {
  KALDI_ASSERT(IsSortedAndUniq(pdfs));
  KALDI_ASSERT(phones != NULL);
  phones->clear();
  for (int32 tstate = 1; tstate <= trans_model.NumTransitionStates(); tstate++) {
    if (std::binary_search(pdfs.begin(), pdfs.end(),
                           trans_model.TransitionStateToForwardPdf(tstate)) ||
        std::binary_search(pdfs.begin(), pdfs.end(),
                           trans_model.TransitionStateToSelfLoopPdf(tstate)))
      phones->push_back(trans_model.TransitionStateToPhone(tstate));
  }
  SortAndUniq(phones);

  for (int32 tstate = 1; tstate <= trans_model.NumTransitionStates(); tstate++)
    if (std::binary_search(phones->begin(), phones->end(),
                           trans_model.TransitionStateToPhone(tstate))
        && !(std::binary_search(pdfs.begin(), pdfs.end(),
                               trans_model.TransitionStateToForwardPdf(tstate)) &&
             std::binary_search(pdfs.begin(), pdfs.end(),
                               trans_model.TransitionStateToSelfLoopPdf(tstate))) )
      return false;
  return true;
}

bool TransitionModel::Compatible(const TransitionModel &other) const {
  return (topo_ == other.topo_ && tuples_ == other.tuples_ &&
          state2id_ == other.state2id_ && id2state_ == other.id2state_
          && num_pdfs_ == other.num_pdfs_);
}

bool TransitionModel::IsSelfLoop(int32 trans_id) const {
  KALDI_ASSERT(static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];
  int32 trans_index = trans_id - state2id_[trans_state];
  const Tuple &tuple = tuples_[trans_state-1];
  int32 phone = tuple.phone, hmm_state = tuple.hmm_state;
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
  KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
  return (static_cast<size_t>(trans_index) < entry[hmm_state].transitions.size()
          && entry[hmm_state].transitions[trans_index].first == hmm_state);
}

} // End namespace kaldi
