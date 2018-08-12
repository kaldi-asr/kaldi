// decoder/grammar-fst.cc

// Copyright   2018  Johns Hopkins University (author: Daniel Povey)

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

#include "grammar-fst.h"

namespace fst {

GrammarFstImpl::GrammarFstImpl(
    const Fst<StdArc> &top_fst,
    const std::vector<std::pair<Label, const Fst<StdArc> *> > &ifsts,
    int32 nonterm_phones_offset):
    top_fst_(top_fst),
    ifst_(fsts),
    nonterm_phones_offset_(nonterm_phones_offset) {
  KALDI_ASSERT(nonterm_phones_offset_ > 1);
  encoding_multiple_ = 1000;
  // the following loop won't be entered in typical system configuration.
  while (encoding_multiple_ <= nonterm_phones_offset_)
    encoding_multiple_ += 1000;
  SetType("grammar");

} // end namespace fst
