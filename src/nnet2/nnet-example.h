// nnet2/nnet-example.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)
//           2014  Vimal Manohar

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
#include "thread/kaldi-semaphore.h"

namespace kaldi {
namespace nnet2 {

/// NnetExample is the input data and corresponding label (or labels) for one
/// or more frames of input, used for standard cross-entropy training of neural
/// nets (and possibly for other objective functions).  In the normal case there
/// will be just one frame, and one label, with a weight of 1.0.
struct NnetExample {

  /// The label(s) for each frame in a sequence of frames; in the normal case,
  /// this will be just [ [ (pdf-id, 1.0) ] ], i.e. one frame with one label.
  /// Top-level index is the frame index; then for each frame, a list of pdf-ids
  /// each with its weight.
  /// In some contexts, we will require that labels.size() == 1.
  std::vector<std::vector<std::pair<int32, BaseFloat> > > labels;  
  
  /// The input data, with NumRows() >= labels.size() + left_context; it
  /// includes features to the left and right as needed for the temporal context
  /// of the network.  The features corresponding to labels[0] would be in
  /// the row with index equal to left_context.
  CompressedMatrix input_frames; 

  /// The number of frames of left context (we can work out the #frames
  /// of right context from input_frames.NumRows(), labels.size(), and this).
  int32 left_context;

  /// The speaker-specific input, if any, or an empty vector if
  /// we're not using this features.  We'll append this to the
  /// features for each of the frames.
  Vector<BaseFloat> spk_info; 
  
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  NnetExample() { }

  /// This constructor can be used to extract one or more frames from an example
  /// that has multiple frames, and possibly truncate the context.  Most of its
  /// behavior is obvious from the variable names, but note the following: if
  /// left_context is -1, we use the left-context of the input; the same for
  /// right_context.  If start_frame < 0 we start the labels from frame 0 of the
  /// labeled frames of ,input; if num_frames == -1 we go to the end of the
  /// labeled input from start_frame.  If start_frame + num_frames is greater
  /// than the number of frames of labels of input, we output as much as we can
  /// instead of crashing.  The same with left_context and right_context-- if we
  /// can't provide the requested context we won't crash but will provide as
  /// much as we can, although in this case we'll print a warning (once).
  NnetExample(const NnetExample &input,
              int32 start_frame,
              int32 num_frames,
              int32 left_context,
              int32 right_context);
  
  /// Set the label of this frame of this example to the specified pdf_id with
  /// the specified weight.
  void SetLabelSingle(int32 frame, int32 pdf_id, BaseFloat weight = 1.0);

  /// Get the maximum weight label (pdf_id and weight) of this frame of this
  /// example.
  int32 GetLabelSingle(int32 frame, BaseFloat *weight = NULL);
};


typedef TableWriter<KaldiObjectHolder<NnetExample > > NnetExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<NnetExample > > SequentialNnetExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<NnetExample > > RandomAccessNnetExampleReader;


/** This class stores neural net training examples to be used in
    multi-threaded training.  */
class ExamplesRepository {
 public:
  /// The following function is called by the code that reads in the examples,
  /// with a batch of examples.  [It will empty the vector "examples").
  void AcceptExamples(std::vector<NnetExample> *examples);

  /// The following function is called by the code that reads in the examples,
  /// when we're done reading examples.
  void ExamplesDone();
  
  /// This function is called by the code that does the training.  It gets the
  /// training examples, and if they are available, puts them in "examples" and
  /// returns true.  It returns false when there are no examples left and
  /// ExamplesDone() has been called.
  bool ProvideExamples(std::vector<NnetExample> *examples);
  
  ExamplesRepository(): empty_semaphore_(1), done_(false) { }
 private:
  Semaphore full_semaphore_;
  Semaphore empty_semaphore_;

  std::vector<NnetExample> examples_;
  bool done_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(ExamplesRepository);
};


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
  

  /// spk_info contains any component of the features that varies slowly or not
  /// at all with time (and hence, we would lose little by averaging it over
  /// time and storing the average).  We'll append this to each of the input
  /// features, if used.
  Vector<BaseFloat> spk_info; 

  void Check() const; // will crash if invalid.
  
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};

// Yes, the length of typenames is getting out of hand.
typedef TableWriter<KaldiObjectHolder<DiscriminativeNnetExample > >
   DiscriminativeNnetExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<DiscriminativeNnetExample > >
   SequentialDiscriminativeNnetExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<DiscriminativeNnetExample > >
   RandomAccessDiscriminativeNnetExampleReader;


}
} // namespace

#endif // KALDI_NNET2_NNET_EXAMPLE_H_
