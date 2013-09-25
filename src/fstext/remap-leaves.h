// fstext/remap-leaves.h

// Copyright 2009-2011  Microsoft Corporation

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

#ifndef KALDI_FSTEXT_REMAP_LEAVES_H_
#define KALDI_FSTEXT_REMAP_LEAVES_H_

/*
   Things in this header relate to transforming an FST whose symbols on the
   left are sequences of phones (as created in context-fst.h) into a
   representation where the individual leaves are on arcs
   (not sequences of leaves).  We also introduce a mapping on the leaves, which
   is intended to allow us to reconstruct the phone sequence (basically, if
   different phones/HMM-positions share a leaf then we introduce new leaf-ids).

   The way that this is intended be used is: create C as a special FST as in
   context-fst.h, create C o L o G (somehow), use functions declared in this
   header to create essentially H o C o L o G (with remapping but with no
   self-loops), and then afterwards determinize and factor the result.  Then
   we will create a wrapper class that "expands" the factored form on the
   fly into a full representation of H o C o L o G with self-loops included
   and possibly with different leaf-ids to disambiguate self-loops from
   forward transitions.  Note also that this wrapper class may also need
   to expand leaf-ids that actually represent HMMs rather than single states,
   as this is a possibility too.
*/


#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include "fstext/context-fst.h"
#include "itf/context-dep-itf.h"


namespace fst {

//! ContextExpandLeaves transforms the sequences of phones on the input of C,
//! into sequences of states.  However it outputs the new symbols as "mangled"
//! state-names that contain sufficient information to obtain the phone ids.
//! ctx_dep [in] object that

template<typename L>
bool ContextExpandLeaves(const kaldi::ContextDependencyInterface &ctx_dep,
                         const vector<L> &phones,
                         const vector<L> &disambig_syms,
                         const vector<vector<L> > &symbol_map_in,
                         vector<vector<L> > *symbol_map_out,
                         vector<L> *aug_to_leaf_out,
                         vector<L> *aug_to_phone_out);




/// DfsSort sorts the states in an FST in depth-first-search order.
/// May be useful after ExpandSequences.  This is equivalent to TopSort
/// for acyclic FSTs but it does apply the DFS order even for FSTs with cycles.
template <class Arc>
void DfsSort(MutableFst<Arc> *fst);

/// The following function doesn't exactly belong here.
/// It creates an FST that transduces the isymbol to its matching osymbols, wherever
/// they share the same string.  It's useful for remapping symbol tables.
/// You will have to explicitly specify the Arc template argument when you call this
/// function, e.g. MakeRemapper<StdArc>(isyms, osyms).
template<class Arc>
VectorFst<Arc> *MakeRemapper(const SymbolTable *isymbols, const SymbolTable *osymbols);



}  // namespace fst

#include "remap-leaves-inl.h"

#endif
