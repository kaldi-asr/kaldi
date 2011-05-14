// fstext/make-stochastic.h

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

#ifndef KALDI_FSTEXT_MAKE_STOCHASTIC_H_
#define KALDI_FSTEXT_MAKE_STOCHASTIC_H_
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <fst/fstlib.h>
#include <fst/fst-decl.h>


namespace fst {



/// MakeStochasticOptions describes the options for the
/// MakeStochasticFst and ReverseMakeStochasticFst functions.
struct MakeStochasticOptions {
  float delta;  // typically equal to kDelta.  For quantization of weights.
  MakeStochasticOptions(): delta(kDelta) { }
};



/**  @brief MakeStochasticFst makes sure the weights out of each arc sum to one,
     and inserts symbols on the output side of the FST to record the leftover
     weights.
     This is done by simply dividing by the current sum, and inserting symbols
     on the output side to "remember" the weight we discarded.  This ensures
     that the FST is stochastic, but it only make sense if the FST was, in an
     appropriate sense, "almost" stochastic to start with.  (For example, a
     language model which, due to backoff arcs, is not properly stochastic).
     You will probably want to cast to the log semiring before calling this
     function.

     If "fst" has an output symbol table, we require that, at input, it should
     not contain any symbols equal to "prefix" plus a digit plus something.
     At output, the symbol table (if it exists) it will contain symbols of the
     form, for example, "##!0.25" for the left-over probabilities.

     @param opts [in] An options parameter that specifies the delta for
            discretization of weights, and a prefix for the numbers [only
            applicable if fst has a symbol table].
     @param fst [in, out] A pointer to the FST that we will change.  If it has a
            symbol table, we will also modify its symbol table.
     @param leftover_probs [out]  This routine will output the
           leftover weights into this vector, indexed by the symbol.  Symbols
           that already existed in the output-symbol table [or on the output
           side of arcs in the FST, if no symbol table, if provided] will have
           a zero entry in "leftover_probs", if they are within the range of
           that vector at output (out-of-range symbols are implicitly zero).
     @param num_symbols_added [out] If non-NULL, this routine outputs to here
           the number of unique symbols that were added to the symbol table.
  */
template<class Arc>
void MakeStochasticFst(const MakeStochasticOptions &opts,
                       MutableFst<Arc> *fst,
                       std::vector<float> *leftover_probs,
                       int *num_symbols_added);


/** This function returns true if, in the semiring of the FST, the sum (within
    the semiring) of all the arcs out of each state in the FST is one, to within
    delta.  After MakeStochasticFst, this should be true (for a connected FST).

    @param fst [in] the FST that we are testing.
    @param delta [in] the tolerance to within which we test equality to 1.
    @param min_sum [out] if non, NULL, contents will be set to the minimum sum of weights.
    @param max_sum [out] if non, NULL, contents will be set to the maximum sum of weights.
    @return Returns true if the FST is stochastic, and false otherwise.
*/

template<class Arc>
bool IsStochasticFst(const Fst<Arc> &fst,
                     float delta = kDelta,  // kDelta = 1.0/1024.0 by default.
                     typename Arc::Weight *min_sum = NULL,
                     typename Arc::Weight *max_sum = NULL);




// IsStochasticFstInLog makes sure it's stochastic after casting to log.
inline bool IsStochasticFstInLog(const VectorFst<StdArc> &fst,
                                 float delta = kDelta,  // kDelta = 1.0/1024.0 by default.
                                 StdArc::Weight *min_sum = NULL,
                                 StdArc::Weight *max_sum = NULL);

/** ReverseMakeStochasticFst does (roughly) the opposite of MakeStochasticFst.
    It removes the "special symbols" that MakeStochasticFst added, and replaces them
    with weights.  However, it does not remove the extra arcs and states that
    MakeStochasticFst added.

    If the "leftover_probs" pointer is non-NULL it uses this vector to work
    out the weights to be added.  Otherwise it works them out from the output
    symbol table provided to the function.  If the "leftover_probs" pointer is
    NULL, the symbol table must be provided and it must have been generated
    by MakeStochasticFst with the same "opts.prefix" value as is in the options
    to the current function call.

   @param [in] opts  Options [we only read the "prefix" element]
   @param [in] leftover_probs  The leftover probabilities are
        looked up from this vector.
   @param [in, out] fst  The fst we are changing.
   @param [out] num_syms_removed  If non-NULL, the function outputs to
        this integer the number of unique symbols that  we extracted the
        probabilities from.   This may be a useful check
        [compare with num_syms_added in MakeStochasticFst].
*/
template<class Arc>
void ReverseMakeStochasticFst(const MakeStochasticOptions &opts,
                              const std::vector<float> &leftover_probs,
                              MutableFst<Arc> *fst,
                              int *num_syms_removed);


/** ShortestDistance is a version of a template that already exists in
    shortest-distance.h, but this one allows you to override the delta.  Assuming
    we are in the log semiring, this function should return Weight::One() after
    MakeStochasticFst.  [the reason we need to be in the log semiring is that
    even if all states are coaccessible and their best cost is one in the
    semiring, they may not have a unit-cost path to the end state.

    This function is useful for testing and double-checking.  E.g. if
    we compose a stochastic FST with a functional unweighted FST,
    the ShortestDistance should still return One().  We may make the
    delta smaller than the default (1/1024.0) for extra accuracy.
*/
template<class Arc>
typename Arc::Weight ShortestDistance(Fst<Arc> &fst, float delta) {
  vector<typename Arc::Weight> distance;
  ReweightType type = REWEIGHT_TO_INITIAL;
  ShortestDistance(fst, &distance, type == REWEIGHT_TO_INITIAL, delta);
  if (fst.Start() == kNoStateId) return Arc::Weight::Zero();
  else return distance[fst.Start()];
}


} // end namespace fst


#include "fstext/make-stochastic-inl.h"

#endif
