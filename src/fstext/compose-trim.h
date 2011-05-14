// fstext/compose-trim.h

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

#ifndef KALDI_FSTEXT_COMPOSE_TRIM_H_
#define KALDI_FSTEXT_COMPOSE_TRIM_H_
#include <fst/fstlib.h>
#include <fst/fst-decl.h>



namespace fst {

/**
   ComposeTrimLeft extracts the subset of "fst1" that contributes to the
   composed FST fst1 o fst2.  It goes through the motions of composing the FSTs
   "fst1" and "fst2", but it does not output the composed result.  Instead it
   uses the information obtained during composition to create "ofst1", which is
   a version of fst1 but with possibly fewer arcs and states and final-weights.
   It trims away everything that does not contribute to the final composed result.

   There are two additional requirements on the semiring and on the fst "fst1".
   For the normal use-case these are not a problem.  These requirements stem
   from the data-structures used internally; it would be possible to devise
   an algorithm that would not be subject to them.  Summary of requirements:
   "tropical semiring OK, log semiring usually OK, gallic semiring not OK."
   The detailed requirements and the reasons for them follow.

   Requirement 1) is that either:
      a) there are no "duplicate arcs", i.e. arcs with identical symbols, weights
         and source-and-dest states.
   or b) the semiring is idempotent (w+w = w for all weights w)

   Requirement (1) just gets rid of a very pathological case which should never
   exist in the first place and is not even supported by the set-theoretic notation
   of Mohri's papers.  The reason is that there is a stage in the algorithm
   where we have duplicate copies of arcs, and we de-duplicate them by comparing
   the arc contents; if there were duplicate arcs in the original FST "fst1", these
   also get de-duplicated.

   Requirement 2) is that;

   The semiring must have a Value() function that returns a type for which the
   default "<" (less than)  operator defines a total order on the underlying weight;
   this eliminates the Gallic semiring.  This is to avoid difficulty in coding-- to
   support Gallic, we could have added a function to it that defines Value() as a
   pair of Value1(), Value2().  This is not really useful as the Gallic semiring is
   typically only used in determinization/minimization, not in composition.  This
   requirement also relates to the way we de-duplicate arcs, since we do so by
   sorting them, and for this we need to define a total order on the arcs.

   @param fst1 [in] The fst on the left hand side of the composition
   @param fst2 [in] The fst on the right hand side of the composition
   @param connect [in] If true [should normally be true], the composition
       algorithm removes non-coacessible states.  This makes the output
       more compact.
   @param ofst1 [out] The output fst which will be a subset of fst1.
*/
template<class Arc>
void ComposeTrimLeft(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
                     bool connect,  MutableFst<Arc> *ofst1);


/** ComposeTrimRight is as ComposeTrimLeft but outputs a trimmed version of its
    right hand argument.

    @param fst1 [in] The fst on the left hand side of the composition
    @param fst2 [in] The fst on the right hand side of the composition
    @param connect [in] If true [should normally be true], the composition
       algorithm removes non-coacessible states.  This makes the output
       more compact.
    @param ofst2 [out] The output fst which will be a subset of fst2.
*/
template<class Arc>
void ComposeTrimRight(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
                      bool connect, MutableFst<Arc> *ofst2);



}

#include "fstext/compose-trim-inl.h"

#endif
