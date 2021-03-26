// nnet3/nnet-chain-example.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
// Copyright      2020  Idiap Research Institute (author: Srikanth Madikeri)

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

#ifndef KALDI_NNET3_NNET_CHAIN_EXAMPLE_H_
#define KALDI_NNET3_NNET_CHAIN_EXAMPLE_H_

#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-computation.h"
#include "hmm/posterior.h"
#include "util/table-types.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"
#include "chain/chain-supervision.h"

namespace kaldi {
namespace nnet3 {


// For regular setups we use struct 'NnetIo' as the output.  For the 'chain'
// models, the output supervision is a little more complex as it involves a
// lattice and we need to do forward-backward, so we use a separate struct for
// it.  The 'output' name means that it pertains to the output of the network,
// as opposed to the features which pertain to the input of the network.  It
// actually stores the lattice-like supervision information at the output of the
// network (which imposes constraints on which frames each phone can be active
// on.
struct NnetChainSupervision {
  /// the name of the output in the neural net; in simple setups it
  /// will just be "output".
  std::string name;

  /// The indexes that the output corresponds to.  The size of this vector will
  /// be equal to supervision.num_sequences * supervision.frames_per_sequence.
  /// Be careful about the order of these indexes-- it is a little confusing.
  /// The indexes in the 'index' vector are ordered as: (frame 0 of each sequence);
  /// (frame 1 of each sequence); and so on.  But in the 'supervision' object,
  /// the FST contains (sequence 0; sequence 1; ...).  So reordering is needed
  /// when doing the numerator computation.
  /// We order 'indexes' in this way for efficiency in the denominator
  /// computation (it helps memory locality), as well as to avoid the need for
  /// the nnet to reorder things internally to match the requested output
  /// (for layers inside the neural net, the ordering is (frame 0; frame 1 ...)
  /// as this corresponds to the order you get when you sort a vector of Index).
  std::vector<Index> indexes;


  /// The supervision object, containing the FST.
  chain::Supervision supervision;

  /// This is a vector of per-frame weights, required to be between 0 and 1,
  /// that is applied to the derivative during training (but not during model
  /// combination, where the derivatives need to agree with the computed objf
  /// values for the optimization code to work).  The reason for this is to more
  /// exactly handle edge effects and to ensure that no frames are
  /// 'double-counted'.  The order of this vector corresponds to the order of
  /// the 'indexes' (i.e. all the first frames, then all the second frames,
  /// etc.)
  /// If this vector is empty it means we're not applying per-frame weights,
  /// so it's equivalent to a vector of all ones.  This vector is written
  /// to disk compactly as unsigned char.
  Vector<BaseFloat> deriv_weights;

  // Use default assignment operator

  NnetChainSupervision() { }

  /// Initialize the object from an object of type chain::Supervision, and some
  /// extra information.  Note: you probably want to set 'name' to "output".
  /// 'first_frame' will often be zero but you can choose (just make it
  /// consistent with how you numbered your inputs), and 'frame_skip' would be 1
  /// in a vanilla setup, but we plan to try setups where the output periodicity
  /// is slower than the input, so in this case it might be 2 or 3.
  NnetChainSupervision(const std::string &name,
                       const chain::Supervision &supervision,
                       const VectorBase<BaseFloat> &deriv_weights,
                       int32 first_frame,
                       int32 frame_skip);

  NnetChainSupervision(const NnetChainSupervision &other);

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  void Swap(NnetChainSupervision *other);

  void CheckDim() const;

  bool operator == (const NnetChainSupervision &other) const;
};

/// NnetChainExample is like NnetExample, but specialized for
/// lattice-free (chain) training.
struct NnetChainExample {

  /// 'inputs' contains the input to the network-- normally just it has just one
  /// element called "input", but there may be others (e.g. one called
  /// "ivector")...  this depends on the setup.
  std::vector<NnetIo> inputs;

  /// 'outputs' contains the chain output supervision.  There will normally
  /// be just one member with name == "output".
  std::vector<NnetChainSupervision> outputs;

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  void Swap(NnetChainExample *other);

  // Compresses the input features (if not compressed)
  void Compress();

  NnetChainExample() { }

  NnetChainExample(const NnetChainExample &other);

  bool operator == (const NnetChainExample &other) const {
    return inputs == other.inputs && outputs == other.outputs;
  }
};

/// This hashing object hashes just the structural aspects of the NnetExample
/// without looking at the value of the features.  It will be used in combining
/// egs into batches of all similar structure.
struct NnetChainExampleStructureHasher {
  size_t operator () (const NnetChainExample &eg) const noexcept;
  // We also provide a version of this that works from pointers.
  size_t operator () (const NnetChainExample *eg) const noexcept {
    return (*this)(*eg);
  }
};


/// This comparator object compares just the structural aspects of the
/// NnetChainExample without looking at the value of the features.
struct NnetChainExampleStructureCompare {
  bool operator () (const NnetChainExample &a,
                    const NnetChainExample &b) const;
  // We also provide a version of this that works from pointers.
  bool operator () (const NnetChainExample *a,
                    const NnetChainExample *b) const {
    return (*this)(*a, *b);
  }
};



/// This function merges a list of NnetChainExample objects into a single one--
/// intended to be used when forming minibatches for neural net training.  If
/// 'compress' it compresses the output features (recommended to save disk
/// space).
///
/// Note: the input is left as it was at the start, but it is temporarily
/// changed inside the function; this is a trick to allow us to use the
/// MergeExamples() routine while avoiding having to rewrite code.
void MergeChainExamples(bool compress,
                        std::vector<NnetChainExample> *input,
                        NnetChainExample *output);



/** Shifts the time-index t of everything in the input of "eg" by adding
    "t_offset" to all "t" values-- but excluding those with names listed in
    "exclude_names", e.g.  "ivector".  This might be useful if you are doing
    subsampling of frames at the output, because shifted examples won't be quite
    equivalent to their non-shifted counterparts.  "exclude_names" is a vector
    of names of nnet inputs that we avoid shifting the "t" values of-- normally
    it will contain just the single string "ivector" because we always leave t=0
    for any ivector.

    Note: input features will be shifted by 'frame_shift', and indexes in the
    supervision in (eg->output) will be shifted by 'frame_shift' rounded to the
    closest multiple of the frame subsampling factor (e.g. 3).  The frame
    subsampling factor is worked out from the time spacing between the indexes
    in the output.  */
void ShiftChainExampleTimes(int32 frame_shift,
                           const std::vector<std::string> &exclude_names,
                           NnetChainExample *eg);

/**  This function takes a NnetChainExample and produces a ComputationRequest.
     Assumes you don't want the derivatives w.r.t. the inputs; if you do, you
     can create the ComputationRequest manually.  Assumes that if
     need_model_derivative is true, you will be supplying derivatives w.r.t. all
     outputs.

     If use_xent_regularization == true, then it assumes that for each output
     name (e.g. "output" in the eg, there is another output with the same
     dimension and with the suffix "-xent" on its name, e.g. named
     "output-xent".  The derivative w.r.t. the xent objective will only be
     supplied to the nnet computation if 'use_xent_derivative' is true (we
     propagate back the xent derivative to the model only in training, not in
     model-combination in nnet3-chain-combine).
*/
void GetChainComputationRequest(const Nnet &nnet,
                                const NnetChainExample &eg,
                                bool need_model_derivative,
                                bool store_component_stats,
                                bool use_xent_regularization,
                                bool use_xent_derivative,
                                ComputationRequest *computation_request);



typedef TableWriter<KaldiObjectHolder<NnetChainExample > > NnetChainExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<NnetChainExample > > SequentialNnetChainExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<NnetChainExample > > RandomAccessNnetChainExampleReader;


/// This function returns the 'size' of a chain example as defined for purposes
/// of merging egs, which is defined as the largest number of Indexes in any of
/// the inputs or outputs of the example.
int32 GetChainNnetExampleSize(const NnetChainExample &a);


/// This class is responsible for arranging examples in groups that have the
/// same strucure (i.e. the same input and output indexes), and outputting them
/// in suitable minibatches as defined by ExampleMergingConfig.
class ChainExampleMerger {
 public:
  ChainExampleMerger(const ExampleMergingConfig &config,
                     NnetChainExampleWriter *writer);

  // This function accepts an example, and if possible, writes a merged example
  // out.  The ownership of the pointer 'a' is transferred to this class when
  // you call this function.
  void AcceptExample(NnetChainExample *a);

  // This function announces to the class that the input has finished, so it
  // should flush out any smaller-sized minibatches, as dictated by the config.
  // This will be called in the destructor, but you can call it explicitly when
  // all the input is done if you want to; it won't repeat anything if called
  // twice.  It also prints the stats.
  void Finish();

  // returns a suitable exit status for a program.
  int32 ExitStatus() { Finish(); return (num_egs_written_ > 0 ? 0 : 1); }

  ~ChainExampleMerger() { Finish(); };
 private:
  // called by Finish() and AcceptExample().  Merges, updates the stats, and
  // writes.  The 'egs' is non-const only because the egs are temporarily
  // changed inside MergeChainEgs.  The pointer 'egs' is still owned
  // by the caller.
  void WriteMinibatch(std::vector<NnetChainExample> *egs);

  bool finished_;
  int32 num_egs_written_;
  const ExampleMergingConfig &config_;
  NnetChainExampleWriter *writer_;
  ExampleMergingStats stats_;

  // Note: the "key" into the egs is the first element of the vector.
  typedef unordered_map<NnetChainExample*,
                        std::vector<NnetChainExample*>,
                        NnetChainExampleStructureHasher,
                        NnetChainExampleStructureCompare> MapType;
MapType eg_to_egs_;
};


bool ParseFromQueryString(const std::string &string,
                          const std::string &key_name,
                          std::string *value);

bool ParseFromQueryString(const std::string &string,
                          const std::string &key_name,
                          BaseFloat *value);

} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_CHAIN_EXAMPLE_H_
