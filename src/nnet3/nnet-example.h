// nnet3/nnet-example.h

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)
//                2014  Vimal Manohar

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

#ifndef KALDI_NNET3_NNET_EXAMPLE_H_
#define KALDI_NNET3_NNET_EXAMPLE_H_

#include "nnet3/nnet-nnet.h"
#include "hmm/posterior.h"
#include "util/table-types.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet3 {


struct NnetIo {
  /// the name of the input in the neural net; in simple setups it
  /// will just be "input".
  std::string name;

  /// "indexes" is a vector the same length as features.NumRows(), explaining
  /// the meaning of each row of the "features" matrix.  Note: the "n" values
  /// in the indexes will always be zero in individual examples, but in general
  /// nonzero after we aggregate the examples into the minibatch level.
  std::vector<Index> indexes;

  /// The features or labels.  GeneralMatrix may contain either a CompressedMatrix,
  /// a Matrix, or SparseMatrix (a SparseMatrix would be the natural format for posteriors).
  GeneralMatrix features;

  /// This constructor creates NnetIo with name "name", indexes with n=0, x=0,
  /// and t values ranging from t_begin to 
  /// (t_begin + t_stride * feats.NumRows() - 1) with a stride t_stride, and
  /// the provided features.  t_begin should be the frame that feats.Row(0)
  /// represents.
  NnetIo(const std::string &name,
         int32 t_begin, const MatrixBase<BaseFloat> &feats,
         int32 t_stride = 1);

  /// This constructor creates NnetIo with name "name", indexes with n=0, x=0,
  /// and t values ranging from t_begin to 
  /// (t_begin + t_stride * feats.NumRows() - 1) with a stride t_stride, and
  /// the provided features.  t_begin should be the frame that the first row
  /// of 'feats' represents.
  NnetIo(const std::string &name,
         int32 t_begin, const GeneralMatrix &feats,
         int32 t_stride = 1);

  /// This constructor sets "name" to the provided string, sets "indexes" with
  /// n=0, x=0, and t from t_begin to (t_begin + t_stride * labels.size() - 1)
  /// with a stride t_stride, and the labels
  /// as provided.  t_begin should be the frame to which labels[0] corresponds.
  NnetIo(const std::string &name,
         int32 dim,
         int32 t_begin,
         const Posterior &labels,
         int32 t_stride = 1);

  void Swap(NnetIo *other);

  NnetIo() { }

  // Use default copy constructor and assignment operators.
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  // this comparison is not very efficient, especially for sparse supervision.
  // It's only used in testing code.
  bool operator == (const NnetIo &other) const;
};


/// This hashing object hashes just the structural aspects of the NnetIo object
/// (name, indexes, feature dimension) without looking at the value of features.
/// It will be used in combining egs into batches of all similar structure.
struct NnetIoStructureHasher {
  size_t operator () (const NnetIo &a) const noexcept;
};
/// This comparison object compares just the structural aspects of the NnetIo
/// object (name, indexes, feature dimension) without looking at the value of
/// features.  It will be used in combining egs into batches of all similar
/// structure.
struct NnetIoStructureCompare {
  bool operator () (const NnetIo &a,
                    const NnetIo &b) const;
};



/// NnetExample is the input data and corresponding label (or labels) for one or
/// more frames of input, used for standard cross-entropy training of neural
/// nets (and possibly for other objective functions).
struct NnetExample {

  /// "io" contains the input and output.  In principle there can be multiple
  /// types of both input and output, with different names.  The order is
  /// irrelevant.
  std::vector<NnetIo> io;

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  NnetExample() { }

  NnetExample(const NnetExample &other): io(other.io) { }

  void Swap(NnetExample *other) { io.swap(other->io); }

  /// Compresses any (input) features that are not sparse.
  void Compress();

  /// Caution: this operator == is not very efficient.  It's only used in
  /// testing code.
  bool operator == (const NnetExample &other) const { return io == other.io; }
};


/// This hashing object hashes just the structural aspects of the NnetExample
/// without looking at the value of the features.  It will be used in combining
/// egs into batches of all similar structure.  Note: the hash value is
/// sensitive to the order in which the NnetIo elements (input and outputs)
/// appear, even though the merging is capable of dealing with
/// differently-ordered inputs and outputs (e.g.  "input" appearing before
/// vs. after "ivector" or "output").  We don't think anyone would ever have to
/// deal with differently-ordered, but otherwise identical, egs in practice so
/// we don't bother making the hashing function independent of this order.
struct NnetExampleStructureHasher {
  size_t operator () (const NnetExample &eg) const noexcept;
  // We also provide a version of this that works from pointers.
  size_t operator () (const NnetExample *eg) const noexcept {
    return (*this)(*eg);
  }
};


/// This comparator object compares just the structural aspects of the
/// NnetExample without looking at the value of the features.  Like
/// NnetExampleStructureHasher, it is sensitive to the order in which the
/// differently-named NnetIo elements appear.  This hashing object will be used
/// in combining egs into batches of all similar structure.
struct NnetExampleStructureCompare {
  bool operator () (const NnetExample &a,
                    const NnetExample &b) const;
  // We also provide a version of this that works from pointers.
  bool operator () (const NnetExample *a,
                    const NnetExample *b) const { return (*this)(*a, *b); }

};



typedef TableWriter<KaldiObjectHolder<NnetExample > > NnetExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<NnetExample > > SequentialNnetExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<NnetExample > > RandomAccessNnetExampleReader;

} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_EXAMPLE_H_
