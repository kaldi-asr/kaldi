// decision-tree/boolean-io-funcs-inl.h

// Copyright 2015  Vimal Manohar

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_DECISION_TREE_BOOLEAN_IO_FUNCS_H_
#define KALDI_DECISION_TREE_BOOLEAN_IO_FUNCS_H_ 1

#include "base/kaldi-common.h"

inline void WriteBooleanVector(std::ostream &os, bool binary,
                               const std::vector<bool> &v);

inline void ReadBooleanVector(std::istream &is, bool binary,
                              std::vector<bool> *v);

#include "decision-tree/boolean-io-funcs-inl.h"

#endif  // KALDI_DECISION_TREE_BOOLEAN_IO_FUNCS_H

