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


struct InputFeature {
  /// the name of the input in the neural net; in simple setups it
  /// will just be "input".
  std::string name;

  /// "indexes" is a vector the same length as features.NumRows(), explaining
  /// the meaning of each row of the "features" matrix.  Note: the "n" values
  /// in the indexes will always be zero here, and are not read/written.
  std::vector<Index> indexes;
  
  /// The features.
  CompressedMatrix features;

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);
};


/// NnetExample is the input data and corresponding label (or labels) for one or
/// more frames of input, used for standard cross-entropy training of neural
/// nets (and possibly for other objective functions). 
struct NnetExample {

  int32 t0;  // time-index corresponding to the first label in the sequence of
             // labels.  Will normally be zero.  [actually the only reason we
             // might want to have it not zero is to train models that are
             // not invariant to time-shift, e.g. clockwork RNNs.
  
  /// "labels" are the labels for each frame in a sequence of frames;it is
  /// indexed first by time-index t = t0 + 0, t0 + 1, .. and then is a list of
  /// (pdf-id, weight).  When training on hard (Viterbi) labels, the normal
  /// case, the inner vectors will have length one, with a single element with
  /// weight 1.0.
  std::vector<std::vector<std::pair<int32, BaseFloat> > > labels;  

  /// some inputs.  Normally there will be just one element in this vector,
  /// with name "input".
  std::vector<InputFeature> features;
  
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  NnetExample(): t0(0) { }

};


typedef TableWriter<KaldiObjectHolder<NnetExample > > NnetExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<NnetExample > > SequentialNnetExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<NnetExample > > RandomAccessNnetExampleReader;

} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_EXAMPLE_H_
