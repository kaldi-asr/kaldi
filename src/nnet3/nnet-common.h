// nnet3/nnet-common.h

// Copyright      2015  Johns Hopkins University (author: Daniel Pove
//                2016  Xiaohui Zhang

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
#include "util/common-utils.h"
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
  bool operator != (const Index &a) const {
    return n != a.n || t != a.t || x != a.x;
  }
  bool operator < (const Index &a) const {
    if (t < a.t) { return true; }
    else if (t > a.t) { return false; }
    else if (x < a.x) { return true; }
    else if (x > a.x) { return false; }
    else return (n < a.n);
  }
  Index operator + (const Index &other) const {
    return Index(n+other.n, t+other.t, x+other.x);
  }
  Index &operator += (const Index &other) {
    n += other.n;
    t += other.t;
    x += other.x;
    return *this;
  }

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &os, bool binary);
};


// this will be the most negative number representable as int32.  It is used as
// the 't' value when we need to mark an 'invalid' index.  This can happen with
// certain non-simple components whose ReorderIndexes() function need to insert
// spaces into their inputs or outputs.
extern const int kNoTime;

// This struct can be used as a comparison object when you want to
// sort the indexes first on n, then x, then t (Index's own comparison
// object will sort first on t, then n, then x)
struct IndexLessNxt {
  inline bool operator ()(const Index &a, const Index &b) const {
    if (a.n < b.n) { return true; }
    else if (a.n > b.n) { return false; }
    else if (a.x < b.x) { return true; }
    else if (a.x > b.x) { return false; }
    else return (a.t < b.t);
  }
};


// this will be used only for debugging output.
std::ostream &operator << (std::ostream &ostream, const Index &index);


void WriteIndexVector(std::ostream &os, bool binary,
                      const std::vector<Index> &vec);

void ReadIndexVector(std::istream &is, bool binary,
                     std::vector<Index> *vec);


/* A Cindex is a pair of a node-index (i.e. the index of a NetworkNode) and an
   Index.  It's frequently used so it gets its own typedef.
 */
typedef std::pair<int32, Index> Cindex;

struct IndexHasher {
  size_t operator () (const Index &cindex) const noexcept;
};

struct CindexHasher {
  size_t operator () (const Cindex &cindex) const noexcept;
};

struct CindexVectorHasher {
  size_t operator () (const std::vector<Cindex> &cindex_vector) const noexcept;
};

// Note: because IndexVectorHasher is used in some things where we really need
// it to be fast, it doesn't look at all the indexes, just most of them.
struct IndexVectorHasher {
  size_t operator () (const std::vector<Index> &index_vector) const noexcept;
};



// this will only be used for pretty-printing.
void PrintCindex(std::ostream &ostream, const Cindex &cindex,
                 const std::vector<std::string> &node_names);

/// this will only be used for pretty-printing.  It prints
/// a vector of Indexes in a compact, human-readable way with
/// compression of ranges (it also doesn't print the x index if it's
/// 1.0.  Example output:
///  "[ (1,1:20), (2, 1:20) ]"
/// which would correspond to the indexes
/// [ (1,1,0), (1,2,0) ... (1,20,0) (2,1,0) ... (2,20,0) ].
void PrintIndexes(std::ostream &ostream,
                  const std::vector<Index> &indexes);

/// this will only be used for pretty-printing.  It prints a vector of Cindexes
/// in a compact, human-readable way with compression of ranges.  If the values
/// of the node indexes are the same for the entire vector, it will just be
/// node-name followed by the output of PrintIndexes, e.g.  some_node[ (1,1,0)
/// ].  Otherwise it will divide the vector into ranges that each have all the
/// same node name, and will print out each range in the way we just mentioned.
/// 'node_names' will usually come from a call like nnet.GetNodeNames().
void PrintCindexes(std::ostream &ostream,
                   const std::vector<Cindex> &cindexes,
                   const std::vector<std::string> &node_names);

/// Appends to 'out' the pairs (node, indexes[0]), (node, indexes[1]), ...
void AppendCindexes(int32 node, const std::vector<Index> &indexes,
                    std::vector<Cindex> *out);

void WriteCindexVector(std::ostream &os, bool binary,
                       const std::vector<Cindex> &vec);

void ReadCindexVector(std::istream &is, bool binary,
                      std::vector<Cindex> *vec);

// this function prints a vector of integers in a human-readable
// way, for pretty-printing; it outputs ranges and repeats in
// a compact form e.g. [ -1x10, 1:20, 25:40 ]
void PrintIntegerVector(std::ostream &ostream,
                        const std::vector<int32> &ints);


// this will be used only for debugging output.
std::ostream &operator << (std::ostream &ostream, const Cindex &cindex);


// some forward declarations.
class Component;
class Nnet;
struct MiscComputationInfo;

} // namespace nnet3
} // namespace kaldi

#endif
