// minimize.h

// Copyright 2010  Microsoft Corporation

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
//
// This is a modified file from the OpenFST Library v1.2.7 available at
// http://www.openfst.org and released under the Apache License Version 2.0.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2005-2010 Google, Inc.
// Author: johans@google.com (Johan Schalkwyk)
//
// \file Functions and classes to minimize a finite state acceptor
//

#ifndef FST_LIB_MINIMIZE_H__
#define FST_LIB_MINIMIZE_H__

#include <cmath>

#include <algorithm>
#include <map>
#include <queue>
#include <vector>
using std::vector;

#include <fst/arcsort.h>
#include <fst/arcmerge.h>
#include <fst/connect.h>
#include <fst/dfs-visit.h>
#include <fst/encode.h>
#include <fst/factor-weight.h>
#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <fst/partition.h>
#include <fst/push.h>
#include <fst/queue.h>
#include <fst/reverse.h>

namespace fst {

// comparator for creating partition based on sorting on
// - states
// - final weight
// - out degree,
// -  (input label, output label, weight, destination_block)
template <class A>
class StateComparator {
 public:
  typedef typename A::StateId StateId;
  typedef typename A::Weight Weight;

  static const uint32 kCompareFinal     = 0x00000001;
  static const uint32 kCompareOutDegree = 0x00000002;
  static const uint32 kCompareArcs      = 0x00000004;
  static const uint32 kCompareAll       = 0x00000007;

  StateComparator(const Fst<A>& fst,
                  const Partition<typename A::StateId>& partition,
                  uint32 flags = kCompareAll)
      : fst_(fst), partition_(partition), flags_(flags) {}

  // compare state x with state y based on sort criteria
  bool operator()(const StateId x, const StateId y) const {
    // check for final state equivalence
    if (flags_ & kCompareFinal) {
      const size_t xfinal = fst_.Final(x).Hash();
      const size_t yfinal = fst_.Final(y).Hash();
      if      (xfinal < yfinal) return true;
      else if (xfinal > yfinal) return false;
    }

    if (flags_ & kCompareOutDegree) {
      // check for # arcs
      if (fst_.NumArcs(x) < fst_.NumArcs(y)) return true;
      if (fst_.NumArcs(x) > fst_.NumArcs(y)) return false;

      if (flags_ & kCompareArcs) {
        // # arcs are equal, check for arc match
        for (ArcIterator<Fst<A> > aiter1(fst_, x), aiter2(fst_, y);
             !aiter1.Done() && !aiter2.Done(); aiter1.Next(), aiter2.Next()) {
          const A& arc1 = aiter1.Value();
          const A& arc2 = aiter2.Value();
          if (arc1.ilabel < arc2.ilabel) return true;
          if (arc1.ilabel > arc2.ilabel) return false;

          if (partition_.class_id(arc1.nextstate) <
              partition_.class_id(arc2.nextstate)) return true;
          if (partition_.class_id(arc1.nextstate) >
              partition_.class_id(arc2.nextstate)) return false;
        }
      }
    }

    return false;
  }

 private:
  const Fst<A>& fst_;
  const Partition<typename A::StateId>& partition_;
  const uint32 flags_;
};

template <class A> const uint32 StateComparator<A>::kCompareFinal;
template <class A> const uint32 StateComparator<A>::kCompareOutDegree;
template <class A> const uint32 StateComparator<A>::kCompareArcs;
template <class A> const uint32 StateComparator<A>::kCompareAll;


// Computes equivalence classes for cyclic Fsts. For cyclic minimization
// we use the classic HopCroft minimization algorithm, which is of
//
//   O(E)log(N),
//
// where E is the number of edges in the machine and N is number of states.
//
// The following paper describes the original algorithm
//  An N Log N algorithm for minimizing states in a finite automaton
//  by John HopCroft, January 1971
//
template <class A, class Queue>
class CyclicMinimizer {
 public:
  typedef typename A::Label Label;
  typedef typename A::StateId StateId;
  typedef typename A::StateId ClassId;
  typedef typename A::Weight Weight;
  typedef ReverseArc<A> RevA;

  CyclicMinimizer(const ExpandedFst<A>& fst):
      // tell the Partition data-member to expect multiple repeated
      // calls to SplitOn with the same element if we are non-deterministic.
      P_(fst.Properties(kIDeterministic, true) == 0) {
    if(fst.Properties(kIDeterministic, true) == 0)
      CHECK(Weight::Properties() & kIdempotent); // this minimization
    // algorithm for non-deterministic FSTs can only work with idempotent
    // semirings.
    Initialize(fst);
    Compute(fst);
  }

  ~CyclicMinimizer() {
    delete aiter_queue_;
  }

  const Partition<StateId>& partition() const {
    return P_;
  }

  // helper classes
 private:
  typedef ArcIterator<Fst<RevA> > ArcIter;
  class ArcIterCompare {
   public:
    ArcIterCompare(const Partition<StateId>& partition)
        : partition_(partition) {}

    ArcIterCompare(const ArcIterCompare& comp)
        : partition_(comp.partition_) {}

    // compare two iterators based on there input labels, and proto state
    // (partition class Ids)
    bool operator()(const ArcIter* x, const ArcIter* y) const {
      const RevA& xarc = x->Value();
      const RevA& yarc = y->Value();
      return (xarc.ilabel > yarc.ilabel);
    }

   private:
    const Partition<StateId>& partition_;
  };

  typedef priority_queue<ArcIter*, vector<ArcIter*>, ArcIterCompare>
  ArcIterQueue;

  // helper methods
 private:
  // prepartitions the space into equivalence classes with
  //   same final weight
  //   same # arcs per state
  //   same outgoing arcs
  void PrePartition(const Fst<A>& fst) {
    VLOG(5) << "PrePartition";

    typedef map<StateId, StateId, StateComparator<A> > EquivalenceMap;
    StateComparator<A> comp(fst, P_, StateComparator<A>::kCompareFinal);
    EquivalenceMap equiv_map(comp);

    StateIterator<Fst<A> > siter(fst);
    StateId class_id = P_.AddClass();
    P_.Add(siter.Value(), class_id);
    equiv_map[siter.Value()] = class_id;
    L_.Enqueue(class_id);
    for (siter.Next(); !siter.Done(); siter.Next()) {
      StateId  s = siter.Value();
      typename EquivalenceMap::const_iterator it = equiv_map.find(s);
      if (it == equiv_map.end()) {
        class_id = P_.AddClass();
        P_.Add(s, class_id);
        equiv_map[s] = class_id;
        L_.Enqueue(class_id);
      } else {
        P_.Add(s, it->second);
        equiv_map[s] = it->second;
      }
    }

    VLOG(5) << "Initial Partition: " << P_.num_classes();
  }

  // - Create inverse transition Tr_ = rev(fst)
  // - loop over states in fst and split on final, creating two blocks
  //   in the partition corresponding to final, non-final
  void Initialize(const Fst<A>& fst) {
    // construct Tr
    Reverse(fst, &Tr_);
    ILabelCompare<RevA> ilabel_comp;
    ArcSort(&Tr_, ilabel_comp);

    // initial split (F, S - F)
    P_.Initialize(Tr_.NumStates() - 1);

    // prep partition
    PrePartition(fst);

    // allocate arc iterator queue
    ArcIterCompare comp(P_);
    aiter_queue_ = new ArcIterQueue(comp);
  }

  // partition all classes with destination C
  void Split(ClassId C) {
    // Prep priority queue. Open arc iterator for each state in C, and
    // insert into priority queue.
    for (PartitionIterator<StateId> siter(P_, C);
         !siter.Done(); siter.Next()) {
      StateId s = siter.Value();
      if (Tr_.NumArcs(s + 1))
        aiter_queue_->push(new ArcIterator<Fst<RevA> >(Tr_, s + 1));
    }

    // Now pop arc iterator from queue, split entering equivalence class
    // re-insert updated iterator into queue.
    Label prev_label = -1;
    while (!aiter_queue_->empty()) {
      ArcIterator<Fst<RevA> >* aiter = aiter_queue_->top();
      aiter_queue_->pop();
      if (aiter->Done()) {
        delete aiter;
        continue;
     }

      const RevA& arc = aiter->Value();
      StateId from_state = aiter->Value().nextstate - 1;
      Label   from_label = arc.ilabel;
      if (prev_label != from_label)
        P_.FinalizeSplit(&L_);

      StateId from_class = P_.class_id(from_state);
      if (P_.class_size(from_class) > 1)
        P_.SplitOn(from_state);

      prev_label = from_label;
      aiter->Next();
      if (aiter->Done())
        delete aiter;
      else
        aiter_queue_->push(aiter);
    }
    P_.FinalizeSplit(&L_);
  }

  // Main loop for hopcroft minimization.
  void Compute(const Fst<A>& fst) {
    // process active classes (FIFO, or FILO)
    while (!L_.Empty()) {
      ClassId C = L_.Head();
      L_.Dequeue();

      // split on C, all labels in C
      Split(C);
    }
  }

  // helper data
 private:
  // Partioning of states into equivalence classes
  Partition<StateId> P_;

  // L = set of active classes to be processed in partition P
  Queue L_;

  // reverse transition function
  VectorFst<RevA> Tr_;

  // Priority queue of open arc iterators for all states in the 'splitter'
  // equivalence class
  ArcIterQueue* aiter_queue_;
};


// Computes equivalence classes for acyclic Fsts. The implementation details
// for this algorithms is documented by the following paper.
//
// Minimization of acyclic deterministic automata in linear time
//  Dominque Revuz
//
// Complexity O(|E|)
//
template <class A>
class AcyclicMinimizer {
 public:
  typedef typename A::Label Label;
  typedef typename A::StateId StateId;
  typedef typename A::StateId ClassId;
  typedef typename A::Weight Weight;

  AcyclicMinimizer(const ExpandedFst<A>& fst):
      // tell the Partition data-member to expect multiple repeated
      // calls to SplitOn with the same element if we are non-deterministic.
      partition_(fst.Properties(kIDeterministic, true) == 0) {
    if(fst.Properties(kIDeterministic, true) == 0)
      CHECK(Weight::Properties() & kIdempotent); // minimization for
    // non-deterministic FSTs can only work with idempotent semirings.
    Initialize(fst);
    Refine(fst);
  }

  const Partition<StateId>& partition() {
    return partition_;
  }

  // helper classes
 private:
  // DFS visitor to compute the height (distance) to final state.
  class HeightVisitor {
   public:
    HeightVisitor() : max_height_(0), num_states_(0) { }

    // invoked before dfs visit
    void InitVisit(const Fst<A>& fst) {}

    // invoked when state is discovered (2nd arg is DFS tree root)
    bool InitState(StateId s, StateId root) {
      // extend height array and initialize height (distance) to 0
      for (size_t i = height_.size(); i <= s; ++i)
        height_.push_back(-1);

      if (s >= num_states_) num_states_ = s + 1;
      return true;
    }

    // invoked when tree arc examined (to undiscoverted state)
    bool TreeArc(StateId s, const A& arc) {
      return true;
    }

    // invoked when back arc examined (to unfinished state)
    bool BackArc(StateId s, const A& arc) {
      return true;
    }

    // invoked when forward or cross arc examined (to finished state)
    bool ForwardOrCrossArc(StateId s, const A& arc) {
      if (height_[arc.nextstate] + 1 > height_[s])
        height_[s] = height_[arc.nextstate] + 1;
      return true;
    }

    // invoked when state finished (parent is kNoStateId for tree root)
    void FinishState(StateId s, StateId parent, const A* parent_arc) {
      if (height_[s] == -1) height_[s] = 0;
      StateId h = height_[s] +  1;
      if (parent >= 0) {
        if (h > height_[parent]) height_[parent] = h;
        if (h > max_height_)     max_height_ = h;
      }
    }

    // invoked after DFS visit
    void FinishVisit() {}

    size_t max_height() const { return max_height_; }

    const vector<StateId>& height() const { return height_; }

    const size_t num_states() const { return num_states_; }

   private:
    vector<StateId> height_;
    size_t max_height_;
    size_t num_states_;
  };

  // helper methods
 private:
  // cluster states according to height (distance to final state)
  void Initialize(const Fst<A>& fst) {
    // compute height (distance to final state)
    HeightVisitor hvisitor;
    DfsVisit(fst, &hvisitor);

    // create initial partition based on height
    partition_.Initialize(hvisitor.num_states());
    partition_.AllocateClasses(hvisitor.max_height() + 1);
    const vector<StateId>& hstates = hvisitor.height();
    for (size_t s = 0; s < hstates.size(); ++s)
      partition_.Add(s, hstates[s]);
  }

  // refine states based on arc sort (out degree, arc equivalence)
  void Refine(const Fst<A>& fst) {
    typedef map<StateId, StateId, StateComparator<A> > EquivalenceMap;
    StateComparator<A> comp(fst, partition_);

    // start with tail (height = 0)
    size_t height = partition_.num_classes();
    for (size_t h = 0; h < height; ++h) {
      EquivalenceMap equiv_classes(comp);

      // sort states within equivalence class
      PartitionIterator<StateId> siter(partition_, h);
      equiv_classes[siter.Value()] = h;
      for (siter.Next(); !siter.Done(); siter.Next()) {
        const StateId s = siter.Value();
        typename EquivalenceMap::const_iterator it = equiv_classes.find(s);
        if (it == equiv_classes.end())
          equiv_classes[s] = partition_.AddClass();
        else
          equiv_classes[s] = it->second;
      }

      // create refined partition
      for (siter.Reset(); !siter.Done();) {
        const StateId s = siter.Value();
        const StateId old_class = partition_.class_id(s);
        const StateId new_class = equiv_classes[s];

        // a move operation can invalidate the iterator, so
        // we first update the iterator to the next element
        // before we move the current element out of the list
        siter.Next();
        if (old_class != new_class)
          partition_.Move(s, new_class);
      }
    }
  }

 private:
  Partition<StateId> partition_;
};


// Given a partition and a mutable fst, merge states of Fst inplace
// (i.e. destructively). Merging works by taking the first state in
// a class of the partition to be the representative state for the class.
// Each arc is then reconnected to this state. All states in the class
// are merged by adding there arcs to the representative state.
template <class A>
void MergeStates(
    const Partition<typename A::StateId>& partition, MutableFst<A>* fst) {
  typedef typename A::StateId StateId;

  vector<StateId> state_map(partition.num_classes());
  for (size_t i = 0; i < partition.num_classes(); ++i) {
    PartitionIterator<StateId> siter(partition, i);
    state_map[i] = siter.Value();  // first state in partition;
  }

  // relabel destination states
  for (size_t c = 0; c < partition.num_classes(); ++c) {
    for (PartitionIterator<StateId> siter(partition, c);
         !siter.Done(); siter.Next()) {
      StateId s = siter.Value();
      for (MutableArcIterator<MutableFst<A> > aiter(fst, s);
           !aiter.Done(); aiter.Next()) {
        A arc = aiter.Value();
        arc.nextstate = state_map[partition.class_id(arc.nextstate)];

        if (s == state_map[c])  // first state just set destination
          aiter.SetValue(arc);
        else
          fst->AddArc(state_map[c], arc);
      }
    }
  }
  fst->SetStart(state_map[partition.class_id(fst->Start())]);

  Connect(fst);
}

template <class A>
void AcceptorMinimize(MutableFst<A>* fst) {
  typedef typename A::StateId StateId;
  if (!(fst->Properties(kAcceptor | kUnweighted, true)))
    LOG(FATAL) << "Input Fst is not an unweighted acceptor";

  // connect fst before minimization, handles disconnected states
  Connect(fst);
  if (fst->NumStates() == 0) return;

  if (fst->Properties(kAcyclic, true)) {
    // Acyclic minimization (revuz)
    VLOG(2) << "Acyclic Minimization";
    ArcSort(fst, ILabelCompare<A>());
    AcyclicMinimizer<A> minimizer(*fst);
    MergeStates(minimizer.partition(), fst);

  } else {
    // Cyclic minimizaton (hopcroft)
    VLOG(2) << "Cyclic Minimization";
    CyclicMinimizer<A, LifoQueue<StateId> > minimizer(*fst);
    MergeStates(minimizer.partition(), fst);
    // Sort arcs before merging
    ArcSort(fst, ILabelCompare<A>());
  }

  // Merge in appropriate semiring
  ArcMerge(fst);
}


// In place minimization of deterministic weighted automata and transducers.
// For transducers, then the 'sfst' argument is not null, the algorithm
// produces a compact factorization of the minimal transducer.
//
// In the acyclic case, we use an algorithm from Dominique Revuz that
// is linear in the number of arcs (edges) in the machine.
//  Complexity = O(E)
//
// In the cyclic case, we use the classical hopcroft minimization.
//  Complexity = O(|E|log(|N|)
//
template <class A>
void Minimize(MutableFst<A>* fst,
              MutableFst<A>* sfst = 0,
              float delta = kDelta) {
  uint64 props = fst->Properties(kAcceptor | kWeighted | kUnweighted, true);
  
  if (!(props & kAcceptor)) {  // weighted transducer
    VectorFst< GallicArc<A, STRING_LEFT> > gfst;
    Map(*fst, &gfst, ToGallicMapper<A, STRING_LEFT>());
    fst->DeleteStates();
    gfst.SetProperties(kAcceptor, kAcceptor);
    Push(&gfst, REWEIGHT_TO_INITIAL, delta);
    Map(&gfst, QuantizeMapper< GallicArc<A, STRING_LEFT> >(delta));
    EncodeMapper< GallicArc<A, STRING_LEFT> >
      encoder(kEncodeLabels | kEncodeWeights, ENCODE);
    Encode(&gfst, &encoder);
    AcceptorMinimize(&gfst);
    Decode(&gfst, encoder);

    if (sfst == 0) {
      FactorWeightFst< GallicArc<A, STRING_LEFT>,
        GallicFactor<typename A::Label,
        typename A::Weight, STRING_LEFT> > fwfst(gfst);
      SymbolTable *osyms = fst->OutputSymbols() ?
          fst->OutputSymbols()->Copy() : 0;
      Map(fwfst, fst, FromGallicMapper<A, STRING_LEFT>());
      fst->SetOutputSymbols(osyms);
      delete osyms;
    } else {
      sfst->SetOutputSymbols(fst->OutputSymbols());
      GallicToNewSymbolsMapper<A, STRING_LEFT> mapper(sfst);
      Map(gfst, fst, &mapper);
      fst->SetOutputSymbols(sfst->InputSymbols());
    }
  } else if (props & kWeighted) {  // weighted acceptor
    Push(fst, REWEIGHT_TO_INITIAL, delta);
    Map(fst, QuantizeMapper<A>(delta));
    EncodeMapper<A> encoder(kEncodeLabels | kEncodeWeights, ENCODE);
    Encode(fst, &encoder);
    AcceptorMinimize(fst);
    Decode(fst, encoder);
  } else {  // unweighted acceptor
    AcceptorMinimize(fst);
  }
}

}  // namespace fst

#endif  // FST_LIB_MINIMIZE_H__
