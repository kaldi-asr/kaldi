// nnet3/nnet-example-utils.h

// Copyright    2015  Johns Hopkins University (author: Daniel Povey)
// Copyright    2020  Idiap Research Institute (author: Srikanth Madikeri)

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
#include "util/kaldi-table.h"

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
    po->Register("num-frames", &num_frames_str, "Number of frames with labels "
                "that each example contains (i.e. the left and right context "
                "are to be added to this).  May just be an integer (e.g. "
                "--num-frames=8), or a principal value followed by "
                "alternative values to be used at most once for each utterance "
                "to deal with odd-sized input, e.g. --num-frames=40,25,50 means "
                "that most of the time the number of frames will be 40, but to "
                "deal with odd-sized inputs we may also generate egs with these "
                "other sizes.  All these values will be rounded up to the "
                "closest multiple of --frame-subsampling-factor.  As a special case, "
                "--num-frames=-1 means 'don't do any splitting'.");
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
  std::vector<BaseFloat> output_weights;
};


class UtteranceSplitter {
 public:

  UtteranceSplitter(const ExampleGenerationConfig &config);


  const ExampleGenerationConfig& Config() const { return config_; }

  // Given an utterance length, this function creates for you a list of chunks
  // into which to split the utterance.  Note: this is partly random (will call
  // srand()).
  // Accumulates some stats which will be printed out in the destructor.
  void GetChunksForUtterance(int32 utterance_length,
                             std::vector<ChunkTimeInfo> *chunk_info);


  // This function returns true if 'supervision_length' (e.g. the length of the
  // posterior, lattice or alignment) is what we expect given
  // config_.frame_subsampling_factor.  If not, it prints a warning (which is
  // why the function needs 'utt', and returns false.  Note: we round up, so
  // writing config_.frame_subsampling_factor as sf, we expect
  // supervision_length = (utterance_length + sf - 1) / sf.
  bool LengthsMatch(const std::string &utt,
                    int32 utterance_length,
                    int32 supervision_length,
                    int32 length_tolerance = 0) const;

  ~UtteranceSplitter();

  int32 ExitStatus() { return (total_frames_in_chunks_ > 0 ? 0 : 1); }

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
  void SetOutputWeights(int32 utterance_length,
                        std::vector<ChunkTimeInfo> *chunk_info) const;

  // Accumulate stats for diagnostics.
  void AccStatsForUtterance(int32 utterance_length,
                            const std::vector<ChunkTimeInfo> &chunk_info);


  const ExampleGenerationConfig &config_;

  // The vector 'splits_for_length_' is indexed by the num-frames of a file, and
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

  // Below are stats used for diagnostics.
  int32 total_num_utterances_;  // total input utterances.
  int64 total_input_frames_;  // total num-frames over all utterances (before
                              // splitting)
  int64 total_frames_overlap_;  // total number of frames that overlap between
                                // adjacent egs.
  int64 total_num_chunks_;
  int64 total_frames_in_chunks_;  // total of chunk-size times count of that
                                  // chunk.  equals the num-frames in all the
                                  // output chunks, added up.
  std::map<int32, int32> chunk_size_to_count_;  // for each chunk size, gives
                                                // the number of chunks with
                                                // that size.

};


class ExampleMergingConfig {
public:
  // The following configuration values are registered on the command line.
  bool compress;
  std::string measure_output_frames;  // for back-compatibility, not used.
  std::string minibatch_size;
  std::string discard_partial_minibatches;   // for back-compatibility, not used.
  bool multilingual_eg; // add language information as a Query (e.g. ?lang=query) to the merged egs's name

  ExampleMergingConfig(const char *default_minibatch_size = "256"):
      compress(false),
      measure_output_frames("deprecated"),
      minibatch_size(default_minibatch_size),
      discard_partial_minibatches("deprecated"),
      multilingual_eg(false)
      { }

  void Register(OptionsItf *po) {
    po->Register("compress", &compress, "If true, compress the output examples "
                 "(not recommended unless you are writing to disk)");
    po->Register("measure-output-frames", &measure_output_frames, "This "
                 "value will be ignored (included for back-compatibility)");
    po->Register("discard-partial-minibatches", &discard_partial_minibatches,
                 "This value will be ignored (included for back-compatibility)");
    po->Register("minibatch-size", &minibatch_size,
                 "String controlling the minibatch size.  May be just an integer, "
                 "meaning a fixed minibatch size (e.g. --minibatch-size=128). "
                 "May be a list of ranges and values, e.g. --minibatch-size=32,64 "
                 "or --minibatch-size=16:32,64,128.  All minibatches will be of "
                 "the largest size until the end of the input is reached; "
                 "then, increasingly smaller sizes will be allowed.  Only egs "
                 "with the same structure (e.g num-frames) are merged.  You may "
                 "specify different minibatch sizes for different sizes of eg "
                 "(defined as the maximum number of Indexes on any input), in "
                 "the format "
                 "--minibatch-size='eg_size1=mb_sizes1/eg_size2=mb_sizes2', e.g. "
                 "--minibatch-size=128=64:128,256/256=32:64,128.  Egs are given "
                 "minibatch-sizes based on the specified eg-size closest to "
                 "their actual size.");
    po->Register("multilingual-eg", &multilingual_eg,
                "Appends language name to the merged egs. Used only by chain2 recipes for now."
                "For example, when merging examples with output-langName we would want to add "
                "?lang=langName");
  }


  // this function computes the derived (private) parameters; it must be called
  // after the command-line parameters are read and before MinibatchSize() is
  // called.
  void ComputeDerived();

  /// This function tells you what minibatch size should be used for this eg.

  ///  @param [in] size_of_eg   The "size" of the eg, as obtained by
  ///                           GetNnetExampleSize() or a similar function (up
  ///                           to the caller).
  ///  @param [in] num_available_egs   The number of egs of this size that are
  ///                            currently available; should be >0.  The
  ///                            value returned will be <= this value, possibly
  ///                            zero.
  ///  @param [in] input_ended   True if the input has ended, false otherwise.
  ///                            This is important because before the input has
  ///                            ended, we will only batch egs into the largest
  ///                            possible minibatch size among the range allowed
  ///                            for that size of eg.
  ///  @return                   Returns the minibatch size to use in this
  ///                            situation, as specified by the configuration.
  int32 MinibatchSize(int32 size_of_eg,
                      int32 num_available_egs,
                      bool input_ended) const;


 private:
  // struct IntSet is a representation of something like 16:32,64, which is a
  // nonempty list of either positive integers or ranges of positive integers.
  // Conceptually it represents a set of positive integers.
  struct IntSet {
    // largest_size is the largest integer in any of the ranges (64 in this
    // example).
    int32 largest_size;
    // e.g. would contain ((16,32), (64,64)) in this example.
    std::vector<std::pair<int32, int32> > ranges;
    // Returns the largest value in any range (i.e. in the set of
    // integers that this struct represents), that is <= max_value,
    // or 0 if there is no value in any range that is <= max_value.
    // In this example, this function would return the following:
    // 128->64, 64->64, 63->32, 31->31, 16->16, 15->0, 0->0
    int32 LargestValueInRange(int32 max_value) const;
  };
  static bool ParseIntSet(const std::string &str, IntSet *int_set);

  // 'rules' is derived from the configuration values above by ComputeDerived(),
  // and are not set directly on the command line.  'rules' is a list of pairs
  // (eg-size, int-set-of-minibatch-sizes); If no explicit eg-sizes were
  // specified on the command line (i.e. there was no '=' sign in the
  // --minibatch-size option), then we just set the int32 to 0.
  std::vector<std::pair<int32, IntSet> > rules;
};


/// This function returns the 'size' of a nnet-example as defined for purposes
/// of merging egs, which is defined as the largest number of Indexes in any of
/// the inputs or outputs of the example.
int32 GetNnetExampleSize(const NnetExample &a);





/// This class is responsible for storing, and displaying in log messages,
/// statistics about how examples of different sizes (c.f. GetNnetExampleSize())
/// were merged into minibatches, and how many examples were left over and
/// discarded.
class ExampleMergingStats {
 public:
  /// Users call this function to inform this class that one minibatch has been
  /// written aggregating 'minibatch_size' separate examples of original size
  /// 'example_size' (e.g. as determined by GetNnetExampleSize(), but the caller
  /// does that.
  /// The 'structure_hash' is provided so that this class can distinguish
  /// between egs that have the same size but different structure.  In the
  /// extremely unlikely eventuality that there is a hash collision, it will
  /// cause misleading stats to be printed out.
  void WroteExample(int32 example_size, size_t structure_hash,
                    int32 minibatch_size);

  /// Users call this function to inform this class that after processing all
  /// the data, for examples of original size 'example_size', 'num_discarded'
  /// examples could not be put into a minibatch and were discarded.
  void DiscardedExamples(int32 example_size, size_t structure_hash,
                         int32 num_discarded);

  /// Calling this will cause a log message with information about the
  /// examples to be printed.
  void PrintStats() const;

 private:
  // this struct stores the stats for examples of a particular size and
  // structure.
  struct StatsForExampleSize {
    int32 num_discarded;
    // maps from minibatch-size (i.e. number of egs that were
    // aggregated into that minibatch), to the number of such
    // minibatches written.
    unordered_map<int32, int32> minibatch_to_num_written;
    StatsForExampleSize(): num_discarded(0) { }
  };


  typedef unordered_map<std::pair<int32, size_t>, StatsForExampleSize,
                        PairHasher<int32, size_t> > StatsType;

  // this maps from a pair (example_size, structure_hash) to to the stats for
  // examples with those characteristics.
  StatsType stats_;

  void PrintAggregateStats() const;
  void PrintSpecificStats() const;

};


/// This class is responsible for arranging examples in groups
/// that have the same strucure (i.e. the same input and output
/// indexes), and outputting them in suitable minibatches
/// as defined by ExampleMergingConfig.
class ExampleMerger {
 public:
  ExampleMerger(const ExampleMergingConfig &config,
                NnetExampleWriter *writer);

  // This function accepts an example, and if possible, writes a merged example
  // out.  The ownership of the pointer 'a' is transferred to this class when
  // you call this function.
  void AcceptExample(NnetExample *a);

  // This function announces to the class that the input has finished, so it
  // should flush out any smaller-sized minibatches, as dictated by the config.
  // This will be called in the destructor, but you can call it explicitly when
  // all the input is done if you want to; it won't repeat anything if called
  // twice.  It also prints the stats.
  void Finish();

  // returns a suitable exit status for a program.
  int32 ExitStatus() { Finish(); return (num_egs_written_ > 0 ? 0 : 1); }

  ~ExampleMerger() { Finish(); };
 private:
  // called by Finish() and AcceptExample().  Merges, updates the
  // stats, and writes.
  void WriteMinibatch(const std::vector<NnetExample> &egs);

  bool finished_;
  int32 num_egs_written_;
  const ExampleMergingConfig &config_;
  NnetExampleWriter *writer_;
  ExampleMergingStats stats_;

  // Note: the "key" into the egs is the first element of the vector.
  typedef unordered_map<NnetExample*, std::vector<NnetExample*>,
                        NnetExampleStructureHasher,
                        NnetExampleStructureCompare> MapType;
   MapType eg_to_egs_;
};

} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_EXAMPLE_UTILS_H_
