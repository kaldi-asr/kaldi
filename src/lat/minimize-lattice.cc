// lat/minimize-lattice.cc

// Copyright 2009-2011  Saarland University (Author: Arnab Ghoshal)
//           2012-2013  Johns Hopkins University (Author: Daniel Povey);  Chao Weng;
//                      Bagher BabaAli
//                2014  Guoguo Chen

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


#include "lat/minimize-lattice.h"
#include "util/stl-utils.h"

namespace fst {

/*
  Process the states in reverse topological order.
  For each state, compute a hash-value that will be the same for states
  that can be combined.  Then for each pair of states with the
  same hash value, check that the "to-states" map to the
  same equivalence class and that the weights are sufficiently similar.
*/
  
template<class Weight, class IntType> class CompactLatticeMinimizer {
 public:
  typedef CompactLatticeWeightTpl<Weight, IntType> CompactWeight;
  typedef ArcTpl<CompactWeight> CompactArc;
  typedef typename CompactArc::StateId StateId;
  typedef typename CompactArc::Label Label;
  typedef size_t HashType;
  
  CompactLatticeMinimizer(MutableFst<CompactArc> *clat,
                          float delta = fst::kDelta):
      clat_(clat), delta_(delta) { }

  bool Minimize() {
    if (clat_->Properties(kTopSorted, true) == 0) {
      if (!TopSort(clat_)) {
        KALDI_WARN << "Topological sorting of state-level lattice failed "
            "(probably your lexicon has empty words or your LM has epsilon cycles; this "
            " is a bad idea.)";
        return false;
      }
    }
    ComputeStateHashValues();
    ComputeStateMap();
    ModifyModel();
    return true;
  }
  
  static HashType ConvertStringToHashValue(const std::vector<IntType> &vec) {
    const HashType prime = 53281;
    kaldi::VectorHasher<IntType> h;
    HashType ans = static_cast<HashType>(h(vec));
    if (ans == 0)  ans = prime;
    // We don't allow a zero answer, as this can cause too many values to be the
    // same.
    return ans;
  }
  
  static void InitHashValue(const CompactWeight &final_weight, HashType *h) {
    const HashType prime1 = 33317, prime2 = 607; // it's pretty random.
    if (final_weight == CompactWeight::Zero()) *h = prime1;
    else *h = prime2 * ConvertStringToHashValue(final_weight.String());
  }

  // It's important that this function and UpdateHashValueForFinalProb be
  // insensitive to the order in which it's called, as the order of the arcs
  // won't necessarily be the same for different equivalent states.
  static void UpdateHashValueForTransition(const CompactWeight &weight,
                                           Label label,
                                           HashType &next_state_hash,
                                           HashType *h) {
    const HashType prime1 = 1447, prime2 = 51907;
    if (label == 0) label = prime2; // Zeros will cause problems.
    *h += prime1 * label *
        (1 + ConvertStringToHashValue(weight.String()) * next_state_hash);
    // Above, the "1 +" is to ensure that if somehow we get zeros due to
    // weird word sequences, they don't propagate.
  }

  void ComputeStateHashValues() {
    // Note: clat_ is topologically sorted, and StateId is
    // signed.  Each state's hash value is only a function of toplogically-later
    // states' hash values.
    state_hashes_.resize(clat_->NumStates());
    for (StateId s = clat_->NumStates() - 1; s >= 0; s--) {
      HashType this_hash;
      InitHashValue(clat_->Final(s), &this_hash);
      for (ArcIterator<MutableFst<CompactArc> > aiter(*clat_, s);
           !aiter.Done(); aiter.Next()) {
        const CompactArc &arc = aiter.Value();
        HashType next_hash;
        if (arc.nextstate > s) {
          next_hash = state_hashes_[arc.nextstate];
        } else {
          KALDI_ASSERT(s == arc.nextstate &&
                       "Lattice not topologically sorted [code error]");
          next_hash = 1;
          KALDI_WARN << "Minimizing lattice with self-loops "
              "(lattices should not have self-loops)";
        }
        UpdateHashValueForTransition(arc.weight, arc.ilabel,
                                     next_hash, &this_hash);
      }
      state_hashes_[s] = this_hash;
    }
  }



  struct EquivalenceSorter {
    // This struct has an operator () which you can interpret as a less-than (<)
    // operator for arcs.  We sort on ilabel; since the lattice is supposed to
    // be deterministic, this should completely determine the ordering (there
    // should not be more than one arc with the same ilabel, out of the same
    // state).  For identical ilabels we next sort on the nextstate, simply to
    // better handle non-deterministic input (we do our best on this, without
    // guaranteeing full minimization).  We could sort on the strings next, but
    // this would be an unnecessary hassle as we only really need good
    // performance on deterministic input.
    bool operator () (const CompactArc &a, const CompactArc &b) const {
      if (a.ilabel < b.ilabel) return true;
      else if (a.ilabel > b.ilabel) return false;
      else if (a.nextstate < b.nextstate) return true;
      else return false;
    }
  };

  
  // This function works out whether s and t are equivalent, assuming
  // we have already partitioned all topologically-later states into
  // equivalence classes (i.e. set up state_map_).
  bool Equivalent(StateId s, StateId t) const {
    if (!ApproxEqual(clat_->Final(s), clat_->Final(t), delta_))
      return false;
    if (clat_->NumArcs(s) != clat_->NumArcs(t))
      return false;
    std::vector<CompactArc> s_arcs;
    std::vector<CompactArc> t_arcs;
    for (int32 iter = 0; iter <= 1; iter++) {
      StateId state = (iter == 0 ? s : t);
      std::vector<CompactArc> &arcs = (iter == 0 ? s_arcs : t_arcs);
      arcs.reserve(clat_->NumArcs(s));
      for (ArcIterator<MutableFst<CompactArc> > aiter(*clat_, state);
           !aiter.Done(); aiter.Next()) {
        CompactArc arc = aiter.Value();
        if (arc.nextstate == state) {
          // This is a special case for states that have self-loops.  If two
          // states have an identical self-loop arc, they may be equivalent.
          arc.nextstate = kNoStateId;
        } else {
          KALDI_ASSERT(arc.nextstate > state);
          //while (state_map_[arc.nextstate] != arc.nextstate)
          arc.nextstate = state_map_[arc.nextstate];
          arcs.push_back(arc);
        }
      }
      EquivalenceSorter s;
      std::sort(arcs.begin(), arcs.end(), s);
    }
    KALDI_ASSERT(s_arcs.size() == t_arcs.size());
    for (size_t i = 0; i < s_arcs.size(); i++) {
      if (s_arcs[i].nextstate != t_arcs[i].nextstate) return false;
      KALDI_ASSERT(s_arcs[i].ilabel == s_arcs[i].olabel); // CompactLattices are
                                                          // supposed to be
                                                          // acceptors.
      if (s_arcs[i].ilabel != t_arcs[i].ilabel) return false;
      // We've already mapped to equivalence classes.
      if (s_arcs[i].nextstate != t_arcs[i].nextstate) return false;
      if (!ApproxEqual(s_arcs[i].weight, t_arcs[i].weight)) return false;
    }
    return true;
  }
  
  void ComputeStateMap() {
    // We have to compute the state mapping in reverse topological order also,
    // since the equivalence test relies on later states being already sorted
    // out into equivalence classes (by state_map_).
    StateId num_states = clat_->NumStates();
    unordered_map<HashType, std::vector<StateId> > hash_groups_;
    
    for (StateId s = 0; s < num_states; s++)
      hash_groups_[state_hashes_[s]].push_back(s);

    state_map_.resize(num_states);
    for (StateId s = 0; s < num_states; s++)
      state_map_[s] = s; // Default mapping.
    

    { // This block is just diagnostic.
      typedef typename unordered_map<HashType,
              std::vector<StateId> >::const_iterator HashIter;
      size_t max_size = 0;
      for (HashIter iter = hash_groups_.begin(); iter != hash_groups_.end();
           ++iter)
        max_size = std::max(max_size, iter->second.size());
      if (max_size > 1000) {
        KALDI_WARN << "Largest equivalence group (using hash) is " << max_size
                   << ", minimization might be slow.";
      }
    }

    for (StateId s = num_states - 1; s >= 0; s--) {
      HashType hash = state_hashes_[s];
      const std::vector<StateId> &equivalence_class = hash_groups_[hash];
      KALDI_ASSERT(!equivalence_class.empty());
      for (size_t i = 0; i < equivalence_class.size(); i++) {
        StateId t = equivalence_class[i];
        // Below, there is no point doing the test if state_map_[t] != t, because
        // in that case we will, before after this, be comparing with another state
        // that is equivalent to t.
        if (t > s && state_map_[t] == t && Equivalent(s, t)) {
          state_map_[s] = t;
          break;
        }
      }
    }
  }

  void ModifyModel() {    
    // Modifies the model according to state_map_;

    StateId num_removed = 0;
    StateId num_states = clat_->NumStates();
    for (StateId s = 0; s < num_states; s++)
      if (state_map_[s] != s)
        num_removed++;
    KALDI_VLOG(3) << "Removing " << num_removed << " of "
                  << num_states << " states.";
    if (num_removed == 0) return; // Nothing to do.
    
    clat_->SetStart(state_map_[clat_->Start()]);

    for (StateId s = 0; s < num_states; s++) {
      if (state_map_[s] != s)
        continue; // There is no point modifying states we're removing.
      for (MutableArcIterator<MutableFst<CompactArc> > aiter(clat_, s);
           !aiter.Done(); aiter.Next()) {
        CompactArc arc = aiter.Value();
        StateId mapped_nextstate = state_map_[arc.nextstate];
        if (mapped_nextstate != arc.nextstate) {
          arc.nextstate = mapped_nextstate;
          aiter.SetValue(arc);
        }
      }
    }
    fst::Connect(clat_);
  }
 private:
  MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *clat_;
  float delta_;
  std::vector<HashType> state_hashes_;
  std::vector<StateId> state_map_; // maps each state to itself or to some
                                   // equivalent state.  Within each equivalence
                                   // class, we pick one arbitrarily.
};

template<class Weight, class IntType>
bool MinimizeCompactLattice(
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *clat,
    float delta) {
  CompactLatticeMinimizer<Weight, IntType> minimizer(clat, delta);
  return minimizer.Minimize();
}

// Instantiate for CompactLattice type.
template
bool MinimizeCompactLattice<kaldi::LatticeWeight, kaldi::int32>(
    MutableFst<kaldi::CompactLatticeArc> *clat, float delta);
  

}  // namespace fst
