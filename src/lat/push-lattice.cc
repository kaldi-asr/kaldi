// lat/push-lattice.cc

// Copyright 2009-2011  Saarland University (Author: Arnab Ghoshal)
//           2012-2013  Johns Hopkins University (Author: Daniel Povey);  Chao Weng;
//                      Bagher BabaAli

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


#include "lat/push-lattice.h"
#include "hmm/transition-model.h"
#include "util/stl-utils.h"

namespace kaldi {


class CompactLatticePusher {  
 public:
  typedef CompactLattice::StateId StateId;
  typedef CompactLatticeArc Arc;
  typedef CompactLatticeWeight Weight;
  
  CompactLatticePusher(CompactLattice *clat): clat_(clat) { }
  bool Push() {
    if (clat_->Properties(fst::kTopSorted, true) == 0) {
      if (!TopSort(clat_)) {
        KALDI_WARN << "Topological sorting of state-level lattice failed "
            "(probably your lexicon has empty words or your LM has epsilon cycles; this "
            " is a bad idea.)";
        return false;
      }
    }
    ComputeShifts();
    ApplyShifts();
    return true;
  }

  // Gets the string of length [end - begin], starting at this
  // state and taking arc "arc_idx" (and thereafter an arbitrary sequence).
  // Note: here, arc_idx == -1 means take an arbitrary path.
  static void GetString(const CompactLattice &clat,
                        CompactLattice::StateId state,
                        size_t arc_idx,
                        std::vector<int32>::iterator begin,
                        std::vector<int32>::iterator end) {
    CompactLatticeWeight final = clat.Final(state);
    size_t len = end - begin;
    KALDI_ASSERT(len >= 0);
    if (len == 0) return;
    if (arc_idx == -1 && final != CompactLatticeWeight::Zero()) {
      const std::vector<int32> &string = final.String();
      KALDI_ASSERT(string.size() >= len &&
                   "Either code error, or paths in lattice have inconsistent lengths");
      std::copy(string.begin(), string.begin() + len, begin);
      return;
    }

    fst::ArcIterator<CompactLattice> aiter(clat, state);
    if (arc_idx != -1)
      aiter.Seek(arc_idx);
    KALDI_ASSERT(!aiter.Done() &&
                 "Either code error, or paths in lattice are inconsistent in length.");

    const Arc &arc = aiter.Value();
    size_t arc_len = arc.weight.String().size();
    if (arc_len >= len) {
      std::copy(arc.weight.String().begin(), arc.weight.String().begin() + len, begin);
    } else {
      std::copy(arc.weight.String().begin(), arc.weight.String().end(), begin);
      // Recurse.
      GetString(clat, arc.nextstate, -1, begin + arc_len, end);
    }
  }

  void CheckForConflict(const Weight &final,
                        StateId state,
                        int32 *shift) {
    // At input, "shift" has the maximum value that we could shift back assuming
    // there is no conflict between the values of the strings.  We need to check
    // if there is conflict, and if so, reduce the "shift".
    bool is_final = (final !=Weight::Zero());
    size_t num_arcs = clat_->NumArcs(state);
    if (num_arcs + (is_final ? 1 : 0) > 1 && shift > 0) {
      // There is potential for conflict between string values, because >1
      // [arc or final-prob].  Find the longest shift up to and including the
      // current shift, that gives no conflict.

      std::vector<int32> string(*shift), compare_string(*shift);
      size_t arc;
      if (is_final) {
        KALDI_ASSERT(final.String().size() >= *shift);
        std::copy(final.String().begin(), final.String().begin() + *shift,
                  string.begin());
        arc = 0;
      } else {
        // set "string" to string if we take 1st arc.
        GetString(*clat_, state, 0, string.begin(), string.end());
        arc = 1;
      }
      for (; arc < num_arcs; arc++) { // for the other arcs..
        GetString(*clat_, state, arc,
                  compare_string.begin(), compare_string.end());
        std::pair<std::vector<int32>::iterator, std::vector<int32>::iterator> pr =
            std::mismatch(string.begin(), string.end(), compare_string.begin());
        if (pr.first != string.end()) { // There was a mismatch.  Reduce the shift
          // to a value where they will match.
          *shift = pr.first - string.begin();
          string.resize(*shift);
          compare_string.resize(*shift);
        }
      }
    }
  }

  void ComputeShifts() {
    StateId num_states = clat_->NumStates();
    shift_vec_.resize(num_states, 0);
    
    // The for loop will only work if StateId is signed, so assert this.
    KALDI_COMPILE_TIME_ASSERT(static_cast<StateId>(-1) < static_cast<StateId>(0) &&
                              "This code only works if StateId is signed; otherwise, fix.");
    // We rely on the topological sorting, so clat_->Start() should be zero or
    // at least any preceding states should be non-accessible.  We leave the
    // shift at zero for the start state because we can't shift to before that.
    for (StateId state = num_states - 1; state > clat_->Start(); state--) {
      size_t num_arcs = clat_->NumArcs(state);
      Weight final = clat_->Final(state);
      if (num_arcs == 0) {
        // we can shift back by the number of transition-ids on the
        // final-prob, if any.
        shift_vec_[state] = final.String().size();
      } else { // We have arcs ...
        int32 shift = std::numeric_limits<int32>::max();
        size_t num_arcs = 0;
        bool is_final = (final != Weight::Zero());
        if (is_final)
          shift = std::min(shift, static_cast<int32>(final.String().size()));
        for (fst::ArcIterator<CompactLattice> aiter(*clat_, state);
             !aiter.Done(); aiter.Next(), num_arcs++) {
          const Arc &arc (aiter.Value());
          shift = std::min(shift, shift_vec_[arc.nextstate] +
                           static_cast<int32>(arc.weight.String().size()));
        }
        CheckForConflict(final, state, &shift);
        shift_vec_[state] = shift;
      }
    }
  }

  void ApplyShifts() {
    StateId num_states = clat_->NumStates();
    for (StateId state = 0; state < num_states; state++) {
      int32 shift = shift_vec_[state];
      std::vector<int32> string;
      for (fst::MutableArcIterator<CompactLattice> aiter(clat_, state);
           !aiter.Done(); aiter.Next()) {
        Arc arc(aiter.Value());
        KALDI_ASSERT(arc.nextstate > state && "Cyclic lattice");

        string = arc.weight.String();
        size_t orig_len = string.size(), next_shift = shift_vec_[arc.nextstate];
        // extend "string" by next_shift.
        string.resize(string.size() + next_shift);
        // The next command sets the last "next_shift" elements of 'string' to
        // the string starting from arc.nextstate (taking an arbitrary path).
        GetString(*clat_, arc.nextstate, -1,
                  string.begin() + orig_len, string.end());
        // Remove the first "shift" elements of this string and set the
        // arc-weight string to this.
        arc.weight.SetString(std::vector<int32>(string.begin() + shift,
                                                string.end()));
        aiter.SetValue(arc);
      }
      
      Weight final = clat_->Final(state);
      if (final != Weight::Zero()) {
        // Erase first "shift" elements of final-prob.
        final.SetString(std::vector<int32>(final.String().begin() + shift,
                                           final.String().end()));
        clat_->SetFinal(state, final);
      }
    }
  }

 private:
  CompactLattice *clat_;

  // For each state s, shift_vec_[s] >= 0  is how much we will shift the
  // transition-ids back at this state.
  std::vector<int32> shift_vec_;
};

bool PushCompactLattice(CompactLattice *clat) {
  CompactLatticePusher pusher(clat);
  return pusher.Push();
}
      


  

}  // namespace kaldi
