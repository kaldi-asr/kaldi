// fstext/compose-special.h

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

#ifndef KALDI_FSTEXT_COMPOSE_SPECIAL_H_
#define KALDI_FSTEXT_COMPOSE_SPECIAL_H_
#include <fst/fstlib.h>
#include <fst/fst-decl.h>



namespace fst {

/**
   ComposeSpecial is a special kind of composition algorithm.  The normal
   composition matches up the output symbols of the left argument (say, FST "A")
   and the input symbols of the right argument (say, FST "B"), and discards the
   "matched" symbols, so the resulting symbols are the inputs of "A" and the outputs
    of "B".  This algorithm, in its most basic version, matches up both input and
    output symbols of "A" and "B" and keeps them both, i.e. they both appear on
    the output of the resulting FST.  In the "matrix" interpretation of FSTs,
    the weight on a successful path with strings X and Y, can be interpreted as an element
    of an infinite matrix with row-index correponding to string X and column
    index corresponding to string Y.  In this interpretation, composition is
    matrix multiplication.  ComposeSpecial, in this interpretation, is
    elementwise product of matrices.

    [note: for this to make sense, I think we have to assume that multiplication
    in this semiring is commutative].
    
    If the output is C, the weight assigned by C to string-pair (X,Y) is
    the product of the weights assigned by A and B to (X,Y).  Let's assume
    C has states corresponding to the Cartesian products of states in (A,B).
    Each path in C will correspond to a (path in A, path in B), that have
    the same string on them.  If we have multiple paths in A and/or B with
    the same string on them, we'll have multiple paths in C too (the product
    of the counts in A and B).  Since multiplication commutes over addition
    in a semiring, we can show that the resulting weight in "C" has the
    right value.

    We only allow sequences in C for which the symbol sequences in A and B match.
    There will be a concept of a filter, whereby we only allow certain orderings
    of the pairs of (state in A, state in B) along the path.  This is to solve
    the same problem as in regular composition, of multiple orderings, except
    it's more complex here as there are 2 symbol-sequences to match up.

    When matching up the state-sequences, we have to ensure that it doesn't
    care which order the input vs. output symbols appear.  (i.e. the syncing
    of the symbols is not supposed to matter).  

    Let's initially have a space of "leftover symbols", to ensure that
    matching can happen even if not synced.  Arbitrarily, we decide that only
    A can have its symbols "waiting".  The state-space will be:

    state in C = (a, b, X, Y)
    where
       a is a state in A,
       b is a state in B
       X is an input-symbol-sequence corresponding to input symbols we consumed in A
         but not yet in B.
       Y is an output-symbol-sequence corresponding to output symbols we consumed in A
         but not yet in B.

    Later we'll have some constraints on the order.
    Initially consider just individual transitions in A or B.

    Transition in A from (a->a') with (x, y) on output.
    (a, b, X, Y) -> (a', b, (X,x), (Y,y)).
    Note that if x or y is epsilon, it just leaves the sequences X and Y unchanged
      (i.e. we don't enter it in the list).
    Transition in B from (b->b') with (x,y) on output.
    (a, b, X, Y) -> (a, b', x\X, y\Y),
     where x\X is X with an initial x removed (or unchanged if x=epsilon), and
     likewise for y\Y.  If either x\X or y\Y is undefined because x or y was
     not the initial symbol of X and/or Y, this transition is disallowed.

    Limiting the number of alternate versions of paths:
      We won't allow states where both X and Y have length > 1.

    Otherwise we'll allow all paths.  This will give us redundant epsilon-transitions,
    but we'll just state that this is only valid for idempotent semirings. [Then
    we can do epsilon-removal and determinization to ensure there are no extra paths].
    
    
 */


/*
  Notes on creating HTK-style lattices.
  Let's make it configurable what to do with silences.

  The options are: include before word; include after word; own-word (you specify
  the label).
  
  You have to specify the lists of phones: inside-word, begin, end, begin-and-end,
    and silence.
    [This is assumed to cover all phones].
  
  we have a struct
    ComputationState
  that as we go along a path in the lattice, updates itself...
  this computation takes in arcs from the original lattice, and outputs
  word-aligned arcs.

  It consists of "stored" ilabels, olabels, and costs.  ComputationState  

 */

struct WordBoundaryInfoOpts {
  void Register(ParseOptions *po) {
  }
}

struct WordBoundaryInfo {
  WordBoundaryInfo(const WordBoundaryInfoOpts &opts); // Initialize from
  // options class.
    
  std::vector<int32> wbegin_phones;
  std::vector<int32> wend_phones;
  std::vector<int32> wbegin_and_end_phones;
  std::vector<int32> winternal_phones;
  std::vector<int32> silence_phones; // e.g. silence, vocalized-noise...
               // if these occur outside a word, we treat them as optional
               // silence.
  int32 silence_label; // The label we give to silence words.
  // (May be zero, but this will give you epsilon arcs in the
  // CompactLattice output).
  int32 partial_word_label; // The label we give to partially
  // formed words that we might get at the end of the utterance
  // if the lattice was "forced out" (no end state was reached).
};

class ComputationState {
  /// Advance the computation state by adding the symbols and weights
  /// from this arc.
  void Advance(const CompactLatticeArc &arc);

  /// If it can output a whole word, it will do so, will put it in arc_out,
  /// and return true; else it will return false.  If it detects an error
  /// condition and *error = false, it will set *error to true and print
  /// a warning.  Example of error
  bool OutputArc(const WordBoundaryInfo &info,
                 CompactLatticeArc *arc_out,
                 bool *error);


  /// FinalWeight() will return "weight" if both transition_ids
  /// and word_labels are empty, otherwise it will return
  /// Weight::Zero().
  fst::TropicalWeight FinalWeight();

  /// This function may be called when you reach the end of
  /// the lattice and this structure hasn't voluntarily
  /// output words using "OutputArc".  You can force it to
  /// output whatever it has remaining, this way.  This will
  /// typically consist of partial words, and this will only
  /// happen for lattices that were somehow broken, i.e.
  /// had not reached the final state.
  bool OutputArcForce(CompactLatticeArc *arc_out,
                      bool *error);
  

  size_t Hash() const;

  bool operator < (const ComputationState &other) const;  

 private:
  std::vector<int32> transition_ids;
  std::vector<int32> word_labels;
  fst::TropicalWeight weight;
};

/// returns true if everything was OK, false if some kind of
/// error was detected (e.g. the words didn't have the kinds of
/// sequences we would expect if the WordBoundaryInfo was
/// correct).  Note: we don't expect silence inside words,
/// or empty words (words with no phones), and we expect
/// the word to start with a wbegin_phone, to end with
/// a wend_phone, and to possibly have winternal_phones
/// inside (or to consist of just one wbegin_and_end_phone).
/// Note: if it returns false, it doesn't mean the lattice
/// that the output is necessarily bad: it might just be that
/// the lattice was "forced out" as the end-state was not
/// reached during decoding, and in this case the output might
/// be usable.
bool WordAlignLattice(const CompactLattice &lat,
                      const WordBoundaryInfo &info,
                      CompactLattice *lat_out);


} // end namespace fst
#endif


