  
#ifndef KALDI_LM_KENLM_FST_H_
#define KALDI_LM_KENLM_FST_H_

#include <base/kaldi-common.h>
#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include <fstext/deterministic-fst.h>
#include "lm/kenlm.h"

namespace kaldi {
/*
  This class wraps KenLM into Kaldi's DeterministicOnDemandFst class,
  so that Kaldi toolkit can now use KenLM as a simple FST language model,
  typical use cases are:
    * big lm decoding
    * lattice rescoring
    * language model shallow fusion(online lm interpolation)
  It serves a very similar role of Kaldi's CARPA,
  with faster query speed and lower memory footprint.
*/

template<class Arc>
class KenLmFst : public fst::DeterministicOnDemandFst<Arc> {
 public:
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename KenLm::State State;
  typedef typename KenLm::WordIndex WordIndex;
 
  explicit KenLmFst(KenLm *lm)
   : lm_(lm), num_states_(0), bos_state_id_(0)
  {
    MapElem elem;
    lm->SetStateToBos(&elem.first);
    elem.second = num_states_;

    std::pair<IterType, bool> ins_result = state_map_.insert(elem);
    KALDI_ASSERT(ins_result.second == true);
    state_vec_.push_back(&ins_result.first->first);
    num_states_++;

    eos_symbol_id_ = lm_->GetSymbolIndex("</s>");
  }
  virtual ~KenLmFst() { }

  virtual StateId Start() { 
    return bos_state_id_;
  }

  virtual bool GetArc(StateId s, Label label, Arc *oarc) {
    KALDI_ASSERT(s < static_cast<StateId>(state_vec_.size()));
    const State* istate = state_vec_[s];
    MapElem e;
    WordIndex word = lm_->GetWordIndex(label);
    BaseFloat log_10_prob = lm_->Score(istate, word, &e.first);
    e.second = num_states_;
    std::pair<IterType, bool> result = state_map_.insert(e);
    if (result.second == true) {
      state_vec_.push_back(&(result.first->first));
      num_states_++;
    }

    oarc->ilabel = label;
    oarc->olabel = oarc->ilabel;
    oarc->nextstate = result.first->second;
    oarc->weight = Weight(-log_10_prob * M_LN10); // KenLM's log10() -> Kaldi's ln()

    return true;
  }

  virtual Weight Final(StateId s) {
    Arc oarc;
    GetArc(s, eos_symbol_id_, &oarc);
    return oarc.weight;
  }

 private:
  KenLm *lm_; // no ownership

 private:
  typedef std::pair<State, StateId> MapElem;
  typedef unordered_map<State, StateId, KenLm::StateHasher> MapType; // use KenLM's MurMurHash as hash function
  typedef typename MapType::iterator IterType;

  MapType state_map_;
  std::vector<const State*> state_vec_;
  StateId num_states_; // state vector index range, [0, num_states_)

  StateId bos_state_id_;  // fst start state id
  WordIndex eos_symbol_id_;

}; // class KenLmFst

} // namespace kaldi
