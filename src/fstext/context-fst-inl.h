// fstext/context-fst-inl.h

// Copyright 2009-2011  Microsoft Corporation;  Jan Silovsky

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

#ifndef KALDI_FSTEXT_CONTEXT_FST_INL_H_
#define KALDI_FSTEXT_CONTEXT_FST_INL_H_
#include "base/kaldi-common.h"
#include "fstext/fstext-utils.h"

// Do not include this file directly.  It is included by context-fst.h.



namespace fst {

/// \addtogroup context_fst_group
/// @{


template<class Arc, class LabelT>
typename ContextFstImpl<Arc, LabelT>::StateId
   ContextFstImpl<Arc, LabelT>::FindState(const vector<LabelT> &seq) {
  // Finds state-id corresponding to this vector of phones.  Inserts it if
  // necessary.
  assert(static_cast<int32>(seq.size()) == N_-1);
  VectorToStateIter iter = state_map_.find(seq);
  if (iter == state_map_.end()) {  // Not already in map.
    StateId this_state_id = (StateId)state_seqs_.size();
    //This check is not needed with OpenFst >= 1.4
#ifndef HAVE_OPENFST_GE_10400
    StateId this_state_id_check = CacheImpl<Arc>::AddState();
    // goes back to VectorFstBaseImpl<Arc>, inherited via CacheFst<Arc>
    assert(this_state_id == this_state_id_check);
#endif
    state_seqs_.push_back(seq);
    state_map_[seq] = this_state_id;
    return this_state_id;
  } else {
    return iter->second;
  }
}

template<class Arc, class LabelT>
typename ContextFstImpl<Arc, LabelT>::Label
ContextFstImpl<Arc, LabelT>::FindLabel(const vector<LabelT> &label_vec) {
  // Finds ilabel corresponding to this information.. Creates new ilabel if necessary.
  VectorToLabelIter iter = ilabel_map_.find(label_vec);
  if (iter == ilabel_map_.end()) {  // Not already in map.
    Label this_label = ilabel_info_.size();
    ilabel_info_.push_back(label_vec);
    ilabel_map_[label_vec] = this_label;
    return this_label;
  } else {
    return iter->second;
  }
}


template<class Arc, class LabelT>
typename ContextFstImpl<Arc, LabelT>::StateId ContextFstImpl<Arc, LabelT>::Start() {
  if (! CacheImpl<Arc>::HasStart()) {
    vector<LabelT> vec(N_-1, 0);  // Vector of N_-1 epsilons. [e.g. N = 3].
    StateId s = FindState(vec);
    assert(s == 0);
    this->SetStart(s);
  }
  return CacheImpl<Arc>::Start();
}



template<class Arc, class LabelT>
ContextFstImpl<Arc, LabelT>::ContextFstImpl(const ContextFstImpl &other):
    phone_syms_(other.phone_syms_),
    disambig_syms_(other.disambig_syms_) {
  KALDI_ERR << "ContextFst copying not yet supported "
            << "[not hard, but would have to test.]";
}


template<class Arc, class LabelT>
ContextFstImpl<Arc, LabelT>::ContextFstImpl(Label subsequential_symbol,  // epsilon not allowed.
                                            const vector<LabelT> &phone_syms,  // on output side of ifst.
                                            const vector<LabelT> &disambig_syms,  // on output
                                            int N,
                                            int P):
    phone_syms_(phone_syms),  disambig_syms_(disambig_syms), subsequential_symbol_(subsequential_symbol) ,
    N_(N), P_(P) {

  {  // This block checks the inputs.
    assert(subsequential_symbol != 0
           && disambig_syms_.count(subsequential_symbol) == 0
           && phone_syms_.count(subsequential_symbol) == 0);
    if (phone_syms.empty())
      KALDI_WARN << "Context FST created but there are no phone symbols: probably input FST was empty.";
    assert(phone_syms_.count(0) == 0);
    assert(disambig_syms_.count(0) == 0);
    for (size_t i = 0; i < phone_syms.size(); i++)
      assert(disambig_syms_.count(phone_syms[i]) == 0);
    assert(N>0 && P>=0 && P<N);
  }
  SetType("context");
  assert(subsequential_symbol_ != 0);  // it's OK to be kNoLabel though, if it never appears in ifst.

  assert(disambig_syms_.count(subsequential_symbol_) == 0 && phone_syms_.count(subsequential_symbol_) == 0);

  vector<LabelT> eps_vec;  // empty vec.
  // Make sure the symbol that equates to epsilon is zero in our numbering.
  Label eps_id = FindLabel(eps_vec);  // this function will add it to the input
  // symbol table, if necessary.
  assert(eps_id == 0);  // doing this in the initializer should guarantee it is zero.

  if (N > P+1 && !disambig_syms_.empty()) {
    // We add in a symbol whose sequence representation is [ 0 ], and whose symbol-id
    // is 1.  This is treated as a disambiguation symbol, we call it #-1 in printed
    // form.  It is necessary to ensure that all determinizable LG's will have determinizable
    // CLG's.  The problem it fixes is quite subtle-- it relates to reordering of
    // disambiguation symbols (they appear earlier in CLG than in LG, relative to phones),
    // and the fact that if a disambig symbol appears at the very start of a sequence in
    // CLG, it's not clear exatly where it appeared on the corresponding sequence at
    // the input of LG.
    vector<LabelT> pseudo_eps_vec;
    pseudo_eps_vec.push_back(0);
    pseudo_eps_symbol_= FindLabel(pseudo_eps_vec);  // this function will add it to the input
    // symbol table, if necessary.
    assert(pseudo_eps_symbol_ == 1);
  } else pseudo_eps_symbol_ = 0;  // use actual epsilon.
}



template<class Arc, class LabelT>
typename ContextFstImpl<Arc, LabelT>::Weight ContextFstImpl<Arc, LabelT>::Final(StateId s) {
  assert(static_cast<size_t>(s) < state_seqs_.size());  // make sure state exists already.
  if (!this->HasFinal(s)) {  // Work out final-state weight.
    const vector<LabelT> &seq = state_seqs_[s];

    bool final_ok;
    assert(static_cast<int32>(seq.size()) == N_-1);

    if (P_ < N_ - 1) {
      /* Note that P_ (in zero based indexing) is the "central position", and for arcs out of
         this state the thing at P_ will be the one we expand.  If this is the subsequential symbol,
         it means we will output nothing (and will obviously never output anything).  Thus we make
         this state the final state.
      */
      final_ok = (seq[P_] == subsequential_symbol_);
    } else {
      /* If P_ == N_-1, then the "central phone" is the last one in the list (we have a left-context system).
         In this case everything is output immediately and there is no need for a subsequential symbol.
         Here, any state in the FST can be the final state.
      */
      final_ok = true;
    }
    Weight w = final_ok ? Weight::One() : Weight::Zero();
    this->SetFinal(s, w);
    return w;
  }
  return CacheImpl<Arc>::Final(s);
}

// Warning!  Not tested for correctness.  Does not really matter, the way
// this function is being used so far.  Note: could possibly be wrong,
template<class Arc, class LabelT>
size_t ContextFstImpl<Arc, LabelT>::NumArcs(StateId s) {
  if (this->HasArcs(s)) {
    return CacheImpl<Arc>::NumArcs(s);
  }
  KALDI_ASSERT(s >= 0 && s < state_seqs_.size());
  const vector<LabelT> &seq = state_seqs_[s];
  KALDI_ASSERT(seq.size() == N_ - 1);
  if (!seq.empty() && seq.back() == subsequential_symbol_) {
    // State is not a "normal" state because it just saw the subsequential symbol,
    // hence it cannot accept phones.

    if (P_ == N_ - 1 || seq[P_] == subsequential_symbol_) { // don't
      // accept subsequential symbol.. c.f. logic in CreateArc().
      return disambig_syms_.size();
    } else {
      return disambig_syms_.size() + 1; // Accept disambig syms and
                                        // subsequential symbol.
    }
  } else {
    // For normal states, in general there is potentially an arc for each phone and an arc
    // for each disambiguation symbol, plus one for the subsequential symbol.
    return phone_syms_.size() + disambig_syms_.size() + 1;
  }
}

template<class Arc, class LabelT>
size_t ContextFstImpl<Arc, LabelT>::NumInputEpsilons(StateId s) {
  if (!this->HasArcs(s))
    Expand(s);
  return CacheImpl<Arc>::NumInputEpsilons(s);
}

template<class Arc, class LabelT>
void ContextFstImpl<Arc, LabelT>::InitArcIterator(StateId s, ArcIteratorData<Arc> *data) {
  if (!this->HasArcs(s))
    Expand(s);
  CacheImpl<Arc>::InitArcIterator(s, data);
}


template<class Arc, class LabelT>
void ContextFstImpl<Arc, LabelT>::CreateDisambigArc(StateId s,
                                                    Label olabel,
                                                    Arc *oarc) {  // called from CreateArc.
  // Creates a self-loop arc corresponding to the disambiguation symbol.
  vector<LabelT> label_info;  // (olabel);
  label_info.push_back(-olabel);  // olabel is a disambiguation symbol.  Use its negative
  // so we can easily distinguish them.
  Label ilabel = FindLabel(label_info);
  oarc->ilabel = ilabel;
  oarc->olabel = olabel;
  oarc->weight = Weight::One();
  oarc->nextstate = s;  // self-loop.
}

template<class Arc, class LabelT>
bool ContextFstImpl<Arc, LabelT>::CreatePhoneOrEpsArc(StateId src,
                                                      StateId dst,
                                                      Label olabel,
                                                      const vector<LabelT> &phone_seq,
                                                      Arc *oarc) {
  // called from CreateArc.
  // creates the arc with a phone's state on its input labels (or epsilon).
  // returns true if it created the arc.
  // returns false if it could not create an arc due to the decision-tree returning false
  // [this only happens if opts_.behavior_on_failure == ContextFstOptions::kNoArc].

  assert(phone_seq[P_] != subsequential_symbol_);  // would be coding error.

  if (phone_seq[P_] == 0) {  // this can happen at the beginning of the graph.
    // we don't output a real phone.  Epsilon arc (but sometimes we need to
    // use a special disambiguation symbol instead of epsilon).
    *oarc = Arc(pseudo_eps_symbol_, olabel, Weight::One(), dst);
    // This 1 is a "special" disambiguation symbol (#-1 in printed form) that
    // we use to represent epsilons.
    return true;
  } else {
    // have a phone in central position.
    Label ilabel = FindLabel(phone_seq);
    *oarc = Arc(ilabel, olabel, Weight::One(), dst);
    return true;
  }
}


// This function is specific to ContextFst.  It's not part of the Fst
// interface but it's called (indirectly)by the special matcher.  It
// attempts to create an arc out of state s, with output label
// "olabel" [it works out the input label from the value of "olabel".
// It returns true if it is able to create an arc, and false
// otherwise.
template<class Arc, class LabelT>
bool ContextFstImpl<Arc, LabelT>::CreateArc(StateId s,
                                            Label olabel,
                                            Arc *oarc) {
  // Returns true to indicate the arc exists.

  if (olabel == 0) return false;  // No epsilon-output arcs in this FST.

  const vector<LabelT> &seq = state_seqs_[s];

  if (IsDisambigSymbol(olabel)) {  // Disambiguation-symbol arcs.. create self-loop.
    CreateDisambigArc(s, olabel, oarc);
    return true;
  } else if (IsPhoneSymbol(olabel) || olabel == subsequential_symbol_) {
    // If all is OK, we shift the old sequence left by 1 and push on the new phone.

    if (olabel != subsequential_symbol_ && !seq.empty() &&
        seq.back() == subsequential_symbol_) {
      return false;  // Phone not allowed to follow subsequential symbol.
    }

    if (olabel == subsequential_symbol_ &&
        (P_ == N_-1 || seq[P_] == subsequential_symbol_)) {
      // We already had "enough" subsequential symbols in a row and don't want to
      // accept any more, or we'd be making the subsequential symbol the central phone.
      return false;
    }

    vector<LabelT> newseq(N_-1);  // seq shifted left by 1.
    for (int i = 0;i < N_-2;i++) newseq[i] = seq[i+1];
    if (N_ > 1) newseq[N_-2] = olabel;

    vector<LabelT> phoneseq(seq);  // copy it before FindState which
    // possibly changes the address.
    StateId nextstate = FindState(newseq);

    phoneseq.push_back(olabel);  // Now it's the full context window of size N_.
    for (int i = 1; i < N_ ; i++)
      if (phoneseq[i] == subsequential_symbol_) phoneseq[i] = 0;  // don't put subseq. symbol on
    // the output arcs, just 0.
    return CreatePhoneOrEpsArc(s, nextstate, olabel, phoneseq, oarc);
  } else {
    KALDI_ERR << "ContextFst: CreateArc, invalid olabel supplied [confusion "
              << "about phone list or disambig symbols?]: " << olabel;
  }
  return false;  // won't get here.  suppress compiler error.
}

// Note that Expand is not called if we do the composition using
// ContextMatcher<Arc>, which is the normal case.
template<class Arc, class LabelT>
void ContextFstImpl<Arc, LabelT>::Expand(StateId s) {  // expands arcs only [not final state weight].
  assert(static_cast<size_t>(s) < state_seqs_.size());  // make sure state exists already.

  // We just try adding all possible symbols on the output side.
  Arc arc;
  if (this->CreateArc(s, subsequential_symbol_, &arc)) {
#ifdef HAVE_OPENFST_GE_10400
    this->PushArc(s, arc);
#else
    this->AddArc(s, arc);
#endif
  }
  for (typename kaldi::ConstIntegerSet<Label>::iterator iter = phone_syms_.begin();
       iter != phone_syms_.end(); ++iter) {
    Label phone = *iter;
    if (this->CreateArc(s, phone, &arc)) {
#ifdef HAVE_OPENFST_GE_10400
      this->PushArc(s, arc);
#else
      this->AddArc(s, arc);
#endif
    }
  }
  for (typename kaldi::ConstIntegerSet<Label>::iterator iter = disambig_syms_.begin();
       iter != disambig_syms_.end(); ++iter) {
    Label disambig_sym = *iter;
    if (this->CreateArc(s, disambig_sym, &arc)) {
#ifdef HAVE_OPENFST_GE_10400
      this->PushArc(s, arc);
#else
      this->AddArc(s, arc);
#endif
    }
  }
  this->SetArcs(s);  // mark the arcs as "done". [so HasArcs returns true].
}


template<class Arc, class LabelT>
ContextFst<Arc, LabelT>::ContextFst(const ContextFst<Arc, LabelT> &fst, bool reset) {
  if (reset) {
    impl_ = new ContextFstImpl<Arc, LabelT>(*(fst.impl_));
    // Copy constructor of ContextFstImpl.
    // Main use of calling with reset = true is to free up memory
    // (e.g. then you could delete original one).  Might be useful in transcription
    // expansion during training.
  } else {
    impl_ = fst.impl_;
    impl_->IncrRefCount();
  }
}



template<class Arc, class LabelT>
bool ContextMatcher<Arc, LabelT>::Find(typename Arc::Label match_label) {
  assert(s_ != kNoStateId);
  // we know at this point that match_type_ == MATCH_OUTPUT.  we are matching output.

  if (match_label == kNoLabel) {
    // A ContextFst has no epsilons on its output.  So
    // we don't need to match self-loops on the other FST.
    ready_ = false;
    return false;
  } else if (match_label == 0) {
    arc_.ilabel = 0;
    arc_.olabel = kNoLabel;  // epsilon_L
    arc_.weight = Weight::One();
    arc_.nextstate = s_;  // loop.
    ready_ = true;
    return true;  // epsilon_L loop.
  } else {
    const ContextFst<Arc, LabelT> *cfst = static_cast<const ContextFst<Arc, LabelT>*> (fst_);  // we checked in initializer, that it is.
    ready_ = cfst->CreateArc(s_, match_label, &arc_);
    return ready_;
  }
}

template<class Arc>
void AddSubsequentialLoop(typename Arc::Label subseq_symbol,
                          MutableFst<Arc> *fst) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  vector<StateId> final_states;
  for (StateIterator<MutableFst<Arc> > siter(*fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    if (fst->Final(s) != Weight::Zero())  final_states.push_back(s);
  }

  StateId superfinal = fst->AddState();
  Arc arc(subseq_symbol, 0, Weight::One(), superfinal);
  fst->AddArc(superfinal, arc);  // loop at superfinal.
  fst->SetFinal(superfinal, Weight::One());

  for (size_t i = 0; i < final_states.size(); i++) {
    StateId s = final_states[i];
    fst->AddArc(s, Arc(subseq_symbol, 0, fst->Final(s), superfinal));
    // No, don't remove the final-weights of the original states..
    // this is so we can add the subsequential loop in cases where
    // there is no context, and it won't hurt.
    // fst->SetFinal(s, Weight::Zero());
    arc.nextstate = final_states[i];
  }
}

template<class I>
void WriteILabelInfo(std::ostream &os, bool binary,
                     const vector<vector<I> > &info) {
  I sz = info.size();
  kaldi::WriteBasicType(os, binary, sz);
  for (I i = 0; i < sz; i++) {
    kaldi::WriteIntegerVector(os, binary, info[i]);
  }
}


template<class I>
void ReadILabelInfo(std::istream &is, bool binary,
                    vector<vector<I> > *info) {
  I sz = info->size();
  kaldi::ReadBasicType(is, binary, &sz);
  assert(info != NULL);
  info->resize(sz);
  for (int i = 0; i < sz; i++) {
    kaldi::ReadIntegerVector(is, binary, &((*info)[i]));
  }
}

// Type I must be signed.
template<class I>
SymbolTable *CreateILabelInfoSymbolTable(const vector<vector<I> > &info,
                                         const SymbolTable &phones_symtab,
                                         std::string separator,
                                         std::string initial_disambig) {  // e.g. separator = "/", initial-disambig="#-1"
  assert(std::numeric_limits<I>::is_signed);  // make sure is signed type.
  assert(!info.empty());
  assert(info[0].empty());
  SymbolTable *ans = new SymbolTable("ilabel-info-symtab");
  int64 s = ans->AddSymbol(phones_symtab.Find(static_cast<int64>(0)));
  assert(s == 0);
  for (size_t i = 1; i < info.size(); i++) {
    if (info[i].size() == 0) {
      KALDI_ERR << "Invalid ilabel-info";
    }
    if (info[i].size() == 1 &&
       info[i][0] <= 0) {
      if (info[i][0] == 0) {  // special symbol at start that we want to call #-1.
        s = ans->AddSymbol(initial_disambig);
        if (s != i) {
          KALDI_ERR << "Disambig symbol " << initial_disambig
                    << " already in vocab";
        }
      } else {
        std::string disambig_sym = phones_symtab.Find(-info[i][0]);
        if (disambig_sym == "") {
          KALDI_ERR << "Disambig symbol " << -info[i][0]
                    << " not in phone symbol-table";
        }
        s = ans->AddSymbol(disambig_sym);
        if (s != i) {
          KALDI_ERR << "Disambig symbol " << disambig_sym
                    << " already in vocab";
        }
      }
    } else {
      // is a phone-context-window.
      std::string newsym;
      for (size_t j = 0; j < info[i].size(); j++) {
        std::string phonesym = phones_symtab.Find(info[i][j]);
        if (phonesym == "") {
          KALDI_ERR << "Symbol " << info[i][j]
                    << " not in phone symbol-table";
        }
        if (j != 0) newsym += separator;
        newsym += phonesym;
      }
      int64 s = ans->AddSymbol(newsym);
      if (s != static_cast<int64>(i)) {
        KALDI_ERR << "Some problem with duplicate symbols";
      }
    }
  }
  return ans;
}

inline void ComposeContext(const vector<int32> &disambig_syms_in,
                           int N, int P,
                           VectorFst<StdArc> *ifst,
                           VectorFst<StdArc> *ofst,
                           vector<vector<int32> > *ilabels_out) {
  assert(ifst != NULL && ofst != NULL);
  assert(N > 0);
  assert(P >= 0);
  assert(P < N);

  vector<int32> disambig_syms(disambig_syms_in);
  std::sort(disambig_syms.begin(), disambig_syms.end());
  vector<int32> all_syms;
  GetInputSymbols(*ifst, false/*no eps*/, &all_syms);
  std::sort(all_syms.begin(), all_syms.end());
  vector<int32> phones;
  for (size_t i = 0; i < all_syms.size(); i++)
    if (!std::binary_search(disambig_syms.begin(),
                            disambig_syms.end(), all_syms[i]))
      phones.push_back(all_syms[i]);

  // Get subsequential symbol that does not clash with
  // any disambiguation symbol or symbol in the FST.
  int32 subseq_sym = 1;
  if (!all_syms.empty())
    subseq_sym = std::max(subseq_sym, all_syms.back() + 1);
  if (!disambig_syms.empty())
    subseq_sym = std::max(subseq_sym, disambig_syms.back() + 1);

  // if P == N-1, it's left-context, and no subsequential symbol needed.
  if (P != N-1)
    AddSubsequentialLoop(subseq_sym, ifst);
  ContextFst<StdArc, int32> cfst(subseq_sym, phones, disambig_syms, N, P);
  ComposeContextFst(cfst, *ifst, ofst);
  *ilabels_out = cfst.ILabelInfo();
}

///

}  // namespace fst



#endif  // KALDI_FSTEXT_CONTEXT_FST_INL_H_
