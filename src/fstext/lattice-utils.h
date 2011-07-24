// fstext/lattice-utils.h

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


#ifndef KALDI_FSTEXT_LATTICE_UTILS_H_
#define KALDI_FSTEXT_LATTICE_UTILS_H_

#include "fst/fstlib.h"
#include "fstext/lattice-weight.h"

namespace fst {


/**
   Convert lattice from a normal FST to a CompactLattice FST.
   This is a bit like converting to the Gallic semiring, except
   the semiring behaves in a different way (designed to take
   the best path).
   Note: the ilabels end up as the symbols on the arcs of the
   output acceptor, and the olabels go to the strings.  To make
   it the other way around (useful for the speech-recognition
   application), set invert=true.
*/
template<class Weight, class Int>
void ConvertLatticeToCompact(
    const ExpandedFst<ArcTpl<Weight> > &ifst,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight,Int> > > *ofst,
    bool invert);

/**
   Convert lattice from a CompactLattice FST to a normal FST.  This is a bit like
   converting from the Gallic semiring.  "ifst" must be an acceptor (i.e.,
   ilabels and olabels should be identical).  If invert=false, the labels on
   "ifst" become the ilabels on "ofst" and the strings in the weights of "ifst"
   becomes the olabels.  If invert=true, this is reversed (useful for speech
   recognition lattices).
*/
template<class Weight, class Int>
void ConvertLatticeFromCompact(
    const ExpandedFst<ArcTpl<CompactLatticeWeightTpl<Weight,Int> > > &ifst,
    MutableFst<ArcTpl<Weight> > *ofst,
    bool invert);

} // end namespace fst

#include "fstext/lattice-utils-inl.h"

#endif  // KALDI_FSTEXT_LATTICE_WEIGHT_H_
