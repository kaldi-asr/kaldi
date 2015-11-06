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

#ifndef KALDI_DECISION_TREE_BOOLEAN_IO_FUNCS_INL_H_
#define KALDI_DECISION_TREE_BOOLEAN_IO_FUNCS_INL_H_ 1

// Do not include this file directly.  
// It is included by decision-tree/boolean-io-funcs.h

#include <limits>
#include <vector>
#include <iterator>

namespace kaldi {

inline void WriteBooleanVector(std::ostream &os, bool binary,
                               const std::vector<bool> &v) {
  if (binary) {
    char sz = sizeof(bool);  // this is currently just a check.
    os.write(&sz, 1);
    std::vector<char> ch_vec;
    std::copy(v.begin(), v.end(), std::back_inserter(ch_vec));
    WriteIntegerVector(os, binary, ch_vec);
  } else {
    // focus here is on prettiness of text form rather than
    // efficiency of reading-in.
    // reading-in is dominated by low-level operations anyway:
    // for efficiency use binary.
    os << "[ ";
    std::vector<bool>::const_iterator iter = v.begin(), end = v.end();
    for (; iter != end; ++iter) {
      WriteBasicType(os, binary, *iter);
    }
    os << "]\n";
  }
  if (os.fail()) {
    throw std::runtime_error("Write failure in WriteBooleanVector.");
  }
}


inline void ReadBooleanVector(std::istream &is, bool binary,
                              std::vector<bool> *v) {
  KALDI_ASSERT(v != NULL);
  if (binary) {
    int sz = is.peek();
    if (sz == sizeof(bool)) {
      is.get();
    } else {  // this is currently just a check.
      KALDI_ERR << "ReadBooleanVector: expected to see type of size "
                << sizeof(bool) << ", saw instead " << sz << ", at file position "
                << is.tellg();
    }
    std::vector<char> ch_vec;
    ReadIntegerVector(is, binary, &ch_vec);
    std::copy(ch_vec.begin(), ch_vec.end(), std::back_inserter(*v));
  } else {
    std::vector<bool> tmp_v;  // use temporary so v doesn't use extra memory
                              // due to resizing.
    is >> std::ws;
    if (is.peek() != static_cast<int>('[')) {
      KALDI_ERR << "ReadBooleanVector: expected to see [, saw "
                << is.peek() << ", at file position " << is.tellg();
    }
    is.get();  // consume the '['.
    is >> std::ws;  // consume whitespace.
    while (is.peek() != static_cast<int>(']')) {
      bool b;
      ReadBasicType(is, binary, &b);
      tmp_v.push_back(b);
      is >> std::ws;  // consume whitespace.
    }
    is.get();  // get the final ']'.
    *v = tmp_v;  
  }
  if (!is.fail()) return;
}

}  // end namespace kaldi.

#endif  // KALDI_DECISION_TREE_BOOLEAN_IO_FUNCS_INL_H

