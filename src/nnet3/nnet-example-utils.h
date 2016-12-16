// nnet3/nnet-example-utils.h

// Copyright    2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_EXAMPLE_UTILS_H_
#define KALDI_NNET3_NNET_EXAMPLE_UTILS_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"

namespace kaldi {
namespace nnet3 {



/** Merge a set of input examples into a single example (typically the size of
    "src" will be the minibatch size).  Will crash if "src" is the empty vector.
    If "compress" is true, it will compress any non-sparse features in the output.
 */
void MergeExamples(const std::vector<NnetExample> &src,
                   bool compress,
                   NnetExample *dest);


/** Shifts the time-index t of everything in the "eg" by adding "t_offset" to
    all "t" values.  This might be useful in things like clockwork RNNs that are
    not invariant to time-shifts, to ensure that we see different shifts of each
    example during training.  "exclude_names" is a vector (not necessarily
    sorted) of names of nnet inputs that we avoid shifting the "t" values of--
    normally it will contain just the single string "ivector" because we always
    leave t=0 for any ivector. */
void ShiftExampleTimes(int32 t_offset,
                       const std::vector<std::string> &exclude_names,
                       NnetExample *eg);

/**  This function takes a NnetExample (which should already have been
     frame-selected, if desired, and merged into a minibatch) and produces a
     ComputationRequest.  It assumes you don't want the derivatives w.r.t. the
     inputs; if you do, you can create/modify the ComputationRequest manually.
     Assumes that if need_model_derivative is true, you will be supplying
     derivatives w.r.t. all outputs.
*/
void GetComputationRequest(const Nnet &nnet,
                           const NnetExample &eg,
                           bool need_model_derivative,
                           bool store_component_stats,
                           ComputationRequest *computation_request);


// Writes as unsigned char a vector 'vec' that is required to have
// values between 0 and 1.
void WriteVectorAsChar(std::ostream &os,
                       bool binary,
                       const VectorBase<BaseFloat> &vec);

// Reads data written by WriteVectorAsChar.
void ReadVectorAsChar(std::istream &is,
                      bool binary,
                      Vector<BaseFloat> *vec);


// Warning: after reading in the values from the command line
// (Register() and then then po.Read()), you should then call ComputeDerived()
// to set up the 'derived values' (parses 'num_frames_str').
struct ExampleExtractionConfig {
  int32 left_context;
  int32 right_context;
  int32 left_context_initial;
  int32 right_context_final;
  int32 num_frames_overlap;
  std::string num_frames_str;


  // The following parameters are derived parameters, computed by
  // ComputeDerived().
  int32 num_frames;  // the 'principal' number of frames
  std::vector<int32> num_frames_alternative;

  ExampleExtractionConfig():
      left_context(0), right_context(0),
      left_context_initial(-1), right_context_initial(-1),
      num_frames_overlap(0),
      num_frames_str("1"), num_frames(-1) { }

  /// This function decodes 'num_frames_str' into 'num_frames' and 'num_frames_alternatives',
  /// and ensures that 'num_frames', and the members of num_frames_alternatives' are
  /// multiples of 'frame_subsampling_factor'.
  ///
  void ComputeDerived();

  void Register(OptionsItf *po) {
    po->Register("left-context", &left_context, "Number of frames of left "
                 "context of input features that are added to each "
                 "example");
    po->Register("right-context", &right_context, "Number of frames of right "
                 "context of input features that are added to each "
                 "example");
    po->Register("left-context-initial", &left_context, "Number of frames "
                 "of left context of input features that are added to each "
                 "example at the start of the utterance (if <0, this "
                 "defaults to the same as --left-context)");
    po->Register("right-context-final", &right_context, "Number of frames "
                 "of right context of input features that are added to each "
                 "example at the end of the utterance (if <0, this "
                 "defaults to the same as --right-context)");
    po->Register("right-context", &right_context, "Number of frames of right "
                 "context of input features that are added to each "
                 "example");
    po->Register("num-frames", &num_frames_str, "Number of frames with labels "
                "that each example contains (i.e. the left and right context "
                "are to be added to this).  May just be an integer (e.g. "
                "--num-frames=8), or an principal value followed by "
                "alternative values to be used at most once for each utterance "
                "to deal with odd-sized input, e.g. --num-frames=40,25,50 means "
                "that most of the time the number of frames will be 40, but to "
                "deal with odd-sized inputs we may also generate egs with these "
                "other sizes.  All these values will be rounded up to the "
                "closest multiple of --frame-subsampling-factor.");
    po.Register("num-frames-overlap", &num_frames_overlap, "Number of frames of "
                "overlap between adjacent examples (advisory, will not be "
                "exactly enforced)");
    po.Register("frame-subsampling-factor", &frame_subsampling_factor, "Used "
                "if the frame-rate of the output labels in the generated "
                "examples will be less than the frame-rate at the input");
  }
};



void ComputeExampleTimeInfo(const ExampleExtractionConfig &config,
                            int32 num_frames_in_utt,

                            SplitIntoRanges(int32 num_frames,
                     int32 frames_per_range,
                     std::vector<int32> *range_starts);



struct ExampleTimeInfo {
  int32 first_frame;
  int32 num_frames;
  int32 left_context;
  int32 right_context;
};


// This function rounds up the quantities 'num_frames' and 'num_frames_overlap'
// to the nearest multiple of the frame_subsampling_factor
void RoundUpNumFrames(int32 frame_subsampling_factor,
                      int32 *num_frames,
                      int32 *num_frames_overlap);





} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_EXAMPLE_UTILS_H_
