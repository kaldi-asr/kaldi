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
struct ExampleGenerationConfig {
  int32 left_context;
  int32 right_context;
  int32 left_context_initial;
  int32 right_context_final;
  int32 num_frames_overlap;
  int32 frame_subsampling_factor;
  std::string num_frames_str;


  // The following parameters are derived parameters, computed by
  // ComputeDerived().

  // the first element of the 'num_frames' vector is the 'principal' number of
  // frames; the remaining elements are alternatives to the principal number of
  // frames, to be used at most once or twice per file.
  std::vector<int32> num_frames;

  ExampleGenerationConfig():
      left_context(0), right_context(0),
      left_context_initial(-1), right_context_final(-1),
      num_frames_overlap(0), frame_subsampling_factor(1),
      num_frames_str("1") { }

  /// This function decodes 'num_frames_str' into 'num_frames', and ensures that
  /// the members of 'num_frames' are multiples of 'frame_subsampling_factor'.
  void ComputeDerived();

  void Register(OptionsItf *po) {
    po->Register("left-context", &left_context, "Number of frames of left "
                 "context of input features that are added to each "
                 "example");
    po->Register("right-context", &right_context, "Number of frames of right "
                 "context of input features that are added to each "
                 "example");
    po->Register("left-context-initial", &left_context_initial, "Number of "
                 "frames of left context of input features that are added to "
                 "each example at the start of the utterance (if <0, this "
                 "defaults to the same as --left-context)");
    po->Register("right-context-final", &right_context_final, "Number of "
                 "frames of right context of input features that are added "
                 "to each example at the end of the utterance (if <0, this "
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
    po->Register("num-frames-overlap", &num_frames_overlap, "Number of frames of "
                 "overlap between adjacent eamples (applies to chunks of size "
                 "equal to the primary [first-listed] --num-frames value... "
                 "will be adjusted for different-sized chunks).  Advisory; "
                 "will not be exactly enforced.");
    po->Register("frame-subsampling-factor", &frame_subsampling_factor, "Used "
                 "if the frame-rate of the output labels in the generated "
                 "examples will be less than the frame-rate at the input");
  }
};



/**
   struct ChunkTimeInfo is used by class UtteranceSplitter to output
   information about how we split an utterance into chunks.
 */

struct ChunkTimeInfo {
  int32 first_frame;
  int32 num_frames;
  int32 left_context;
  int32 right_context;
  // The 'output_weights' member is a vector of length equal to the
  // num_frames divided by frame_subsampling_factor from the config.
  // It contains values 0 < x <= 1 that represent weightings of
  // output-frames.  The idea is that if (because of overlaps) a
  // frame appears in multiple chunks, we want to downweight it
  // so that the total weight remains 1.  (Of course, the calling
  // code is free to ignore these weights if desired).
  std::vector<float> output_weights;
};


class UtteranceSplitter {
 public:

  UtteranceSplitter(const ExampleGenerationConfig &config);


  const ExampleGenerationConfig& Config() const { return config_; }

  // Given an utterance length, this function creates for you a list of chunks
  // into which to split the utterance.  Note: this is partly random (will call
  // srand()).
  void GetChunksForUtterance(int32 utterance_length,
                             std::vector<ChunkTimeInfo> *chunk_info) const;


  // This function returns true if 'supervision_length' (e.g. the length of the
  // posterior, lattice or alignment) is what we expect given
  // config_.frame_subsampling_factor.  If not, it prints a warning (which is
  // why the function needs 'utt', and returns false.  Note: we round up, so
  // writing config_.frame_subsampling_factor as sf, we expect
  // supervision_length = (utterance_length + sf - 1) / sf.
  bool LengthsMatch(const std::string &utt,
                    int32 utterance_length,
                    int32 supervision_length) const;


 private:


  void InitSplitForLength();

  // This function returns the 'default duration' in frames of a split, which if
  // config_.num_frames_overlap is zero is just the sum of chunk sizes in the
  // split (i.e. the sum of the vector's elements), but otherwise, we subtract
  // the recommended overlap (see code for details).
  float DefaultDurationOfSplit(const std::vector<int32> &split) const;


  // Used in InitSplitForLength(), returns the maximum utterance-length considered
  // separately in split_for_length_.  [above this, we'll assume that the additional
  // length is consumed by multiples of the 'principal' chunk size.]  It returns
  // the primary chunk-size (config_.num_frames[0]) plus twice the largest of
  // any of the allowed chunk sizes (i.e. the max of config_.num_frames)
  int32 MaxUtteranceLength() const;

  // Used in InitSplitForLength(), this function outputs the set of allowed
  // splits, represented as a sorted list of nonempty vectors (each split is a
  // sorted list of chunk-sizes).
  void InitSplits(std::vector<std::vector<int32> > *splits) const;


  // Used in GetChunksForUtterance, this function selects the list of
  // chunk-sizes for that utterance (later on, the positions and and left/right
  // context information for the chunks will be added to this).  We don't call
  // this a 'split', although it's also a list of chunk-sizes, because we
  // randomize the order in which the chunk sizes appear, whereas for a 'split'
  // we sort the chunk-sizes because a 'split' is conceptually an
  // order-independent representation.
  void GetChunkSizesForUtterance(int32 utterance_length,
                                 std::vector<int32> *chunk_sizes) const;


  // Used in GetChunksForUtterance, this function selects the 'gap sizes'
  // before each of the chunks.  These 'gap sizes' may be positive (representing
  // a gap between chunks, or a number of frames at the beginning of the file that
  // don't correspond to a chunk), or may be negative, corresponding to overlaps
  // between adjacent chunks.
  //
  // If config_.frame_subsampling_factor > 1 and enforce_subsampling_factor is
  // true, this function will ensure that all elements of 'gap_sizes' are
  // multiples of config_.frame_subsampling_factor.  (we always enforce this,
  // but we set it to false inside a recursion when we recurse).  Note: if
  // config_.frame_subsampling_factor > 1, it's possible for the last chunk to
  // go over 'utterance_length' by up to config_.frame_subsampling_factor - 1
  // frames (i.e. it would require that many frames past the utterance end).
  // This will be dealt with when generating egs, by duplicating the last frame.
  void GetGapSizes(int32 utterance_length,
                   bool enforce_subsampling_factor,
                   const std::vector<int32> &chunk_sizes,
                   std::vector<int32> *gap_sizes) const;


  // this static function, used in GetGapSizes(), writes random values to a
  // vector 'vec' such the sum of those values equals n (n may be positive or
  // negative).  It tries to make those values as similar as possible (they will
  // differ by at most one), and the location of the larger versus smaller
  // values is random. 'vec' must be nonempty.
  static void DistributeRandomlyUniform(int32 n,
                                        std::vector<int32> *vec);

  // this static function, used in GetGapSizes(), writes values to a vector
  // 'vec' such the sum of those values equals n (n may be positive or
  // negative).  It tries to make those values, as exactly as it can,
  // proportional to the values in 'magnitudes', which must be positive.  'vec'
  // must be nonempty, and 'magnitudes' must be the same size as 'vec'.
  static void DistributeRandomly(int32 n,
                                 const std::vector<int32> &magnitudes,
                                 std::vector<int32> *vec);

  // This function is responsible for setting the 'output_weights'
  // members of the chunks.
  void SetOutputWeights(int32 utterance_lengths,
                        std::vector<ChunkTimeInfo> *chunk_info) const;


  const ExampleGenerationConfig &config_;

  // The vector 'split_for_length_' is indexed by the num-frames of a file, and
  // gives us a list of alternative splits that we can use if the utternace has
  // that many frames.  For example, if split_for_length[100] = ( (25, 40, 40),
  // (40, 65) ), it means we could either split as chunks of size (25, 40, 40)
  // or as (40, 65).  (we'll later randomize the order).  should use one chunk
  // of size 25 and two chunks of size 40.  In general these won't add up to
  // exactly the length of the utterance; we'll have them overlap (or have small
  // gaps between them) to account for this, and the details of this will be
  // randomly decided per file.  If splits_for_length_[u] is empty, it means the
  // utterance was shorter than the smallest possible chunk size, so
  // we will have to discard the utterance.

  // If an utterance's num-frames is >= split_for_length.size(), the way to find
  // the split to use is to keep subtracting the primary num-frames (==
  // config_.num_frames[0]) minus the num-frames-overlap, from the utterance
  // length, until the resulting num-frames is < split_for_length_.size(),
  // chunks, and then add the subtracted number of copies of the primary
  // num-frames to the split.
  std::vector<std::vector<std::vector<int32> > > splits_for_length_;


};




} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_EXAMPLE_UTILS_H_
