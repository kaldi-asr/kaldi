// fstext/reorder-inl.h

// Copyright 2009-2011  Microsoft Corporation

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

#ifndef KALDI_FSTEXT_REORDER_INL_H_
#define KALDI_FSTEXT_REORDER_INL_H_

#include "util/stl-utils.h"
#include "fstext/factor.h"
// Do not include this file directly.  It is included by reorder.h.

namespace fst {


// Compare class for comparing likelihoods on arcs.
// Makes sense for LogArc and StdArc and similar.

template<class A> class ProbCompare {
 public:
  bool operator() (const A &arc1, const A &arc2) const {
    // The less-than operator on Value() corresponds to a more-than
    // operator on the weights themselves, and this means that
    // the arcs will be in reverse order by weight (greater weights
    // first).
    return arc1.weight.Value() < arc2.weight.Value();
  }

  uint64 Properties(uint64 props) const {
    return (props & kArcSortProperties);
  }
};

inline void WeightArcSort(MutableFst<StdArc> *fst) {
  ProbCompare<StdArc> comp;
  ArcSort(fst, comp);
}

inline void WeightArcSort(MutableFst<LogArc> *fst) {
  ProbCompare<LogArc> comp;
  ArcSort(fst, comp);
}

void DfsReorder(const Fst<StdArc> &fst, MutableFst<StdArc> *ofst) {
  typedef StdArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;
  typedef Arc::Weight Weight;

  ofst->DeleteStates();
  if (fst.Start() < 0) return;  // empty FST.
  vector<StateId> order;
  DfsOrderVisitor<Arc> dfs_order_visitor(&order);
  DfsVisit(fst, &dfs_order_visitor);
  assert(order.size() > 0);
  StateId max_state = *(std::max_element(order.begin(), order.end()));
  vector<StateId> new_number(max_state+1, kNoStateId);
  for (size_t i = 0; i < order.size(); i++)
    new_number[order[i]] = i;
  for (size_t i = 0; i < order.size(); i++) ofst->AddState();
  ofst->SetStart(0);
  for (StateId s = 0; s < static_cast<StateId>(order.size()); s++) {
    StateId old_s = order[s];
    for (ArcIterator<Fst<StdArc> > aiter(fst, old_s);
        !aiter.Done();
        aiter.Next()) {
      Arc arc = aiter.Value();
      StateId old_nextstate = arc.nextstate;
      assert(static_cast<size_t>(old_nextstate) < new_number.size());
      arc.nextstate = new_number[old_nextstate];
      assert(arc.nextstate != kNoStateId);
      ofst->AddArc(s, arc);
    }
    ofst->SetFinal(s, fst.Final(old_s));
  }
}

} // end namespace fst.

#endif
