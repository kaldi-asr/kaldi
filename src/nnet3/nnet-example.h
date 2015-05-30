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
#include "util/table-types.h"
#include "lat/kaldi-lattice.h"
#include "thread/kaldi-semaphore.h"

namespace kaldi {
namespace nnet3 {


struct Feature {
  /// the name of the input in the neural net; in simple setups it
  /// will just be "input".
  std::string name;

  /// "indexes" is a vector the same length as features.NumRows(), explaining
  /// the meaning of each row of the "features" matrix.  Note: the "n" values
  /// in the indexes will always be zero in individual examples, but in general
  /// nonzero after we aggregate the examples into the minibatch level.
  std::vector<Index> indexes;
  
  /// The features.  We store them as PossiblyCompressedMatrix to easily support
  /// turning the compression on and off.
  PossiblyCompressedMatrix features;

  /// This constructor creates Feature with name "name", indexes with n=0, x=0,
  /// and t values ranging from t_begin to t_begin + feats.NumRows() - 1, and
  /// the provided features.  t_begin should be the frame that feats.Row(0)
  /// represents.
  Feature(const std::string &name,
          int32 t_begin, const MatrixBase<BaseFloat> &feats);
  
  // Use default copy constructor and assignment operators.

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);
};


struct Supervision {
  /// the name of the output of the neural net; in simple setups it will just be
  /// "output", but in multi-task learning there could be multiple outputs.
  std::string name;
  
  /// "indexes" is a vector the same length as "labels", explaining
  /// the meaning of each element of "labels".
  std::vector<Index> indexes;

  /// each labels[i] is a list of (label, weight) pairs; in the normal case it
  /// will contain just a single element, with weight 1.0.  this vector has the
  /// same size sa "indexes", which explains which frame each label corresponds
  /// to.
  /// Note: this is the same type as typedef "Posterior".
  std::vector<std::vector<std::pair<int32, BaseFloat> > > labels;

  /// This constructor sets "name" to the provided string, sets "indexes" with
  /// n=0, x=0, and t from t_begin to t_begin + labels.size() - 1, and the labels
  /// as provided.  t_begin should be the frame to which labels[0] corresponds.
  Supervision(const std::string &name,
              int32 t_begin,
              const std::vector<std::vector<std::pair<int32, BaseFloat> > > &labels);
  
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  // Returns true if for each i, labels[i].size() == 1.
  bool HasSimpleLabels() const;
  
};


/// SupervisionVec is a vector of Supervision, to contain supervision for
/// possibly multiple outputs of the neural net, although in typical setups this
/// vector will have just one element.
struct SupervisionVec {
  std::vector<Supervision> supervision;
  
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);
};


/// NnetExample is the input data and corresponding label (or labels) for one or
/// more frames of input, used for standard cross-entropy training of neural
/// nets (and possibly for other objective functions). 
struct NnetExample {

  /// "input" contains the features.  In principle there can be multiple types
  /// of feature with different names.
  std::vector<Feature> input;

  /// "supervision" contains the labels.
  std::vector<Supervision> supervision;
  
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  // Compress any features that are not currently compressed.
  void Compress();
  
  NnetExample() { }

  NnetExample(const NnetExample &other);
  
};


typedef TableWriter<KaldiObjectHolder<NnetExample > > NnetExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<NnetExample > > SequentialNnetExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<NnetExample > > RandomAccessNnetExampleReader;

} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_EXAMPLE_H_
