// nnet3/nnet-cctc-example.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_CCTC_EXAMPLE_H_
#define KALDI_NNET3_NNET_CCTC_EXAMPLE_H_

#include "nnet3/nnet-nnet.h"
#include "hmm/posterior.h"
#include "util/table-types.h"
#include "nnet3/nnet-example.h"
#include "ctc/cctc-supervision.h"
#include "ctc/cctc-training.h"

namespace kaldi {
namespace nnet3 {


// For regular setups we use struct 'NnetIo' as the output.  For CTC, the output
// supervision is a little more complex as it involves a lattice and we need to
// do forward-backward, so we use a separate struct for it.  The 'output' name
// means that it pertains to the output of the network, as opposed to features
// which pertain to the input of the network.  It actually stores the
// lattice-like supervision information at the output of the network (which
// imposes constraints on which frames each phone can be active on.
struct NnetCctcSupervision {
  /// the name of the output in the neural net; in simple setups it
  /// will just be "output".
  std::string name;

  /// The indexes that the output corresponds to.  The size of this vector
  /// will be the sum of the 'num_frames' values of the elements of 'supervision'.
  std::vector<Index> indexes;

  /// This is a vector of CctcSupervision.  When we first aggregate examples we
  /// will have one for each individual example, but then we'll call Compactify(),
  /// and successive CctcSupervision objects that have the same weight will get
  /// merged.  This will make the GPU computation more efficient.
  std::vector<ctc::CctcSupervision> supervision;


  /** This function, which will be called between the Forward and Backward
      passes of training, accepts the neural net output, does the forward-backward
      computation, and computes the derivatives.

      @param [in] opts   The CCTC training options (for normalizing_weight and min_post)
      @param [in] cctc_trans_model   Certain essential indexes for CTC that we
                  put in one place, along with the phone-language-model info.
      @param [in] cu_weights   Obtained from cctc_trans_model.GetWeights().
      @param [in] nnet_output   The output of the neural net (its num-rows
                  should equal indexes->size() (where 'indexes' is the
                  output of GetIndexes()), and its num-cols should equal
                  cctc_trans_model.NumOutputIndexes().
      @param [out] tot_weight The total weight of this training data, used for
                  correctly normalizing the objective function (e.g. for
                  display), is output to here.  It will be set to the sum of
                  supervision[i].weight * supervision[i].num_frames.
      @param [out] tot_objf  The total objective function for this training
                  data, i.e. the sum of the log-likelihoods from the forward
                  computation over each of the supervision examples.
      @param [out] nnet_out_deriv  If non-NULL, the derivative of the CCTC
                  objective function w.r.t. 'nnet_output' will be written to
                  here.  Does not have to be zeroed beforehand (this function
                  does that for you).  Must (if non-NULL) have same dim as
                  nnet_output.
  */
  void ComputeObjfAndDerivs(const ctc::CctcTrainingOptions &opts,
                            const ctc::CctcTransitionModel &cctc_trans_model,
                            const CuMatrix<BaseFloat> &cu_weights,
                            const CuMatrixBase<BaseFloat> &nnet_output,
                            BaseFloat *tot_weight,
                            BaseFloat *tot_objf,
                            CuMatrixBase<BaseFloat> *nnet_out_deriv) const;
  

  // Use default assignment operator

  NnetCctcSupervision();

  /// Initialize the object from an object of type CctcSupervision, and some
  /// extra information.
  /// Note: you probably want to set 'name' to "output".
  /// 'first_frame' will often be zero but you can choose (just make it consistent
  /// with how you numbered your inputs), and 'frame_skip' would be 1 in a
  /// vanilla setup, but we plan to try setups where the output periodicity
  /// is slower than the input, so in this case it might be 2 or 3.
  NnetCctcSupervision(const ctc::CctcSupervision &ctc_supervision,
                      const std::string &name,
                      int32 first_frame,
                      int32 frame_skip);
  
  NnetCctcSupervision(const NnetCctcSupervision &other);
  
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

};

/// NnetCctcExample is like NnetExample, but specialized for CTC training.
/// (actually CCTC training, which is our extension of CTC).
struct NnetCctcExample {

  /// 'inputs' contains the input to the network-- normally just it has just one
  /// element called "input", but there may be others (e.g. one called
  /// "ivector")...  this depends on the setup.
  std::vector<NnetIo> inputs;

  /// 'outputs' contains the CTC output supervision.  There will normally
  /// be just one member with name == "output".
  std::vector<NnetCctcSupervision> outputs;

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  void Swap(NnetCctcExample *other);

  // Compresses the input features (if not compressed)
  void Compress();

  NnetCctcExample() { }

  NnetCctcExample(const NnetCctcExample &other);
};


/// This function merges a list of NnetCctcExample objects into a single one--
/// intended to be used when forming minibatches for neural net training.  If
/// 'compress' it compresses the output features (recommended to save disk
/// space); if 'compactify' is true, it compactifies the output by merging the
/// CctcSupervision objects (recommended for efficiency).
///
/// Note: if you are calling this from multi-threaded code, be aware that
/// internally this function temporarily changes 'input' by means of a
/// const_cast, before putting it back to its original value.  This is a trick
/// to allow us to use the MergeExamples() routine and avoid having to rewrite
/// code.
void MergeCctcExamples(const std::vector<NnetCctcExample> &input,
                      bool compress,
                      bool compactify,
                      NnetCctcExample *output);



typedef TableWriter<KaldiObjectHolder<NnetCctcExample > > NnetCctcExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<NnetCctcExample > > SequentialNnetCctcExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<NnetCctcExample > > RandomAccessNnetCctcExampleReader;

} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_EXAMPLE_H_
