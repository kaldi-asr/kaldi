// fstext/reorder.h

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

#ifndef KALDI_FSTEXT_REORDER_H_
#define KALDI_FSTEXT_REORDER_H_



#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include "util/const-integer-set.h"

namespace fst {



/// Sorts the arcs within each state in decreasing order of weight,
/// in the semiring, i.e. from the most to least probable arcs.
inline void WeightArcSort(Fst<StdArc> *fst);

/// Sorts the arcs within each state in decreasing order of weight,
/// in the semiring, i.e. from the most to least probable arcs.
inline void WeightArcSort(Fst<LogArc> *fst);

/// Reorder states according to depth-first order (ensures that linear chains
/// are successive).  For best results, call it after WeightArcSort.
inline void DfsReorder(const Fst<StdArc> &fst, MutableFst<StdArc> *ofst);

}  // namespace fst

#include "reorder-inl.h"

#endif
