// nnet3/nnet-common.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)


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

#ifndef KALDI_NNET3_NNET_COMMON_H_
#define KALDI_NNET3_NNET_COMMON_H_

#include "base/kaldi-common.h"
#include "itf/options-itf.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix-lib.h"

#include <iostream>

namespace kaldi {
namespace nnet3 {


/**
   struct Index is intended to represent the various indexes by which we number
   the rows of the matrices that the Components process: mainly 'n', the index
   of the member of the minibatch, 't', used for the frame index in speech
   recognition, and 'x', which is a catch-all extra index which we might use in
   convolutional setups or for other reasons.  It is possible to extend this by
   adding new indexes if needed. 
*/
struct Index {
  int32 n;  // member-index of minibatch, or zero.
  int32 t;  // time-frame.
  int32 x;  // this may come in useful in convoluational approaches.
  // ... it is possible to add extra index here, if needed.
  Index(): n(0), t(0), x(0) { }
  Index(int32 n, int32 t, int32 x = 0): n(n), t(t), x(x) { }
  
  bool operator == (const Index &a) const {
    return n == a.n && t == a.t && x == a.x;
  }
  bool operator < (const Index &a) const {
    if (n < a.n) { return true; }
    else if (n > a.n) { return false; }
    else if (t < a.t) { return true; }
    else if (t > a.t) { return false; }
    else return (x < a.x);
  }
  Index operator + (const Index &other) const {
    return Index(n+other.n, t+other.t, x+other.x);
  }
};

/* A Cindex is a pair of a node-index (i.e. the index of a NetworkNode) and an
   Index.  It's frequently used so it gets its own typedef.
 */
typedef std::pair<int32, Index> Cindex;

struct CindexHasher {
  size_t operator () (const Cindex &cindex) const;
};

// some forward declarations.
class Component;
class Nnet;
struct MiscComputationInfo;

#endif
