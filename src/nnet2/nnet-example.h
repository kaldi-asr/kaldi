// nnet2/nnet-example.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_NNET_EXAMPLE_H_
#define KALDI_NNET2_NNET_EXAMPLE_H_

#include "nnet2/nnet-nnet.h"
#include "util/table-types.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {
namespace nnet2 {

// NnetExample is the input data and corresponding labels (or labels)
// for one frame of input, used for standard cross-entropy training of neural
// nets.  In the normal case there will be just one label, with a weight of 1.0.
// But, for example, in discriminative training there might be a mixture of
// labels with different weights.  (note: we may not end up using
// this for discriminative training after all.)
struct NnetExample {

  /// The label(s) for this frame; in the normal case, this will be a vector of
  /// length one, containing (the pdf-id, 1.0)
  std::vector<std::pair<int32, BaseFloat> > labels;  

  /// The input data-- typically with NumRows() more than
  /// labels.size(), it includes features to the left and
  /// right as needed for the temporal context of the network.
  /// (see the left_context variable).
  CompressedMatrix input_frames; 

  /// The number of frames of left context (we can work out the #frames
  /// of right context from input_frames.NumRows(), labels.size(), and this).
  int32 left_context;


  /// The speaker-specific input, if any, or an empty vector if
  /// we're not using this features.  We'll append this to each of the
  Vector<BaseFloat> spk_info; 
  
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};


typedef TableWriter<KaldiObjectHolder<NnetExample > > NnetExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<NnetExample > > SequentialNnetExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<NnetExample > > RandomAccessNnetExampleReader;


/**
   This struct is used to store the information we need for discriminative training
   (MMI or MPE).  Each example corresponds to one chunk of a file (for better randomization
   and to prevent instability, we may split files in the middle).
   The example contains the numerator alignment, the denominator lattice, and the
   input features (extended at the edges according to the left-context and right-context
   the network needs).  It may also contain a speaker-vector (note: this is
   not part of any standard recipe right now but is included in case it's useful
   in the future).
 */
struct DiscriminativeNnetExample {
  /// The weight we assign to this example;
  /// this will typically be one, but we include it
  /// for the sake of generality.  
  BaseFloat weight; 

  /// The numerator alignment
  std::vector<int32> num_ali; 

  /// The denominator lattice.  Note: any acoustic
  /// likelihoods in the denominator lattice will be
  /// recomputed at the time we train.
  CompactLattice den_lat; 

  /// The input data-- typically with a number of frames [NumRows()] larger than
  /// labels.size(), because it includes features to the left and right as
  /// needed for the temporal context of the network.  (see also the
  /// left_context variable).
  /// Caution: when we write this to disk, we do so as CompressedMatrix.
  /// Because we do various manipulations on these things in memory, such
  /// as splitting, we don't want it to be a CompressedMatrix in memory
  /// as this would be wasteful in time and also would lead to further loss of
  /// accuracy.
  Matrix<BaseFloat> input_frames;

  /// The number of frames of left context in the features (we can work out the
  /// #frames of right context from input_frames.NumRows(), num_ali.size(), and
  /// this).
  int32 left_context;
  

  /// The speaker-specific input, if any, or an empty vector if
  /// we're not using this features.  We'll append this to each of the
  /// input features, if used.
  Vector<BaseFloat> spk_info; 

  void Check() const; // will crash if invalid.
  
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};

// Tes, the length of typenames is getting out of hand.
typedef TableWriter<KaldiObjectHolder<DiscriminativeNnetExample > >
   DiscriminativeNnetExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<DiscriminativeNnetExample > >
   SequentialDiscriminativeNnetExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<DiscriminativeNnetExample > >
   RandomAccessDiscriminativeNnetExampleReader;


}
} // namespace

#endif // KALDI_NNET2_NNET_EXAMPLE_H_
