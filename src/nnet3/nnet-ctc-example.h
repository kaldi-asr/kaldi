// nnet3/nnet-ctc-example.h

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

#ifndef KALDI_NNET3_NNET_CTC_EXAMPLE_H_
#define KALDI_NNET3_NNET_CTC_EXAMPLE_H_

#include "nnet3/nnet-nnet.h"
#include "hmm/posterior.h"
#include "util/table-types.h"
#include "nnet3/nnet-example.h"
#include "ctc/ctc-supervision.h"
#include "ctc/cctc-training.h"

namespace kaldi {
namespace nnet3 {


// For regular setups we use struct 'NnetIo' as the output.  For CTC, the output
// supervision is a little more complex as it involves a lattice and we need to
// do forward-backward, so we use a separate struct for it.
struct NnetCtcOutput {
  /// the name of the output in the neural net; in simple setups it
  /// will just be "output".
  std::string name;

  /// This is a vector of CtcSupervision, indexed by 'n' value.  The length of
  /// this vector will equal the number of distinct sequences in the minibatch
  /// (e.g. 256).  The assumption is that we cover 'n' values that start from 0,
  /// without gaps.
  std::vector<ctc::CtcSupervision> supervision;

  /// This function works out the indexes corresponding to the output, from the
  /// 'supervision'.  The output indexes will be ordered as first n=0, then
  /// for n=1, and so on.  The 'x' indexes will always be zero.
  void GetIndexes(std::vector<Index> *indexes) const;

  /// This function, which will be called between the Forward and Backward
  /// passes of training, accepts the neural net output, does the forward-backward
  /// computation, and computes the derivatives.
  /// The nnet output is interpreted as coming in chunks, the first chunk for
  /// the first CtcSupervision (for n=0), and so on.  The way the indexes are
  /// computed by the GetIndexes() function is how we tell the nnet3 code
  /// that this is the case.
  void ComputeObjfAndDerivatives(const ctc::CctcTrainingOptions &opts,
                                 const ctc::CctcTransitionModel &trans_model,
                                 const CuMatrix<BaseFloat> &cu_weights,
                                 const ctc::CtcSupervision &supervision,
                                 const CuMatrixBase<BaseFloat> &nnet_output,
                                 CuMatrix<BaseFloat> *nnet_output_deriv) const;
  

  // Use default copy constructor and assignment operators.
  
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

};


/// NnetCtcExample is like NnetExample, but specialized for CTC training.
/// (actually CCTC training, which is our extension of CTC).
struct NnetCtcExample {

  /// 'inputs' contains the input to the network-- normally just it has just one
  /// element called "input", but there may be others (e.g. one called
  /// "ivector")...  this depends on the setup.
  std::vector<NnetIo> inputs;

  /// 'outputs' contains the CTC output supervision.  There will normally
  /// be just one member with name = "output".
  std::vector<NnetCtcOutput> outputs;

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  void Swap(NnetCtcExample *other);

  // Compress the input features (if not compressed).
  void Compress();

  NnetCtcExample() { }

  NnetCtcExample(const NnetCtcExample &other);
};


typedef TableWriter<KaldiObjectHolder<NnetCtcExample > > NnetCtcExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<NnetCtcExample > > SequentialNnetCtcExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<NnetCtcExample > > RandomAccessNnetCtcExampleReader;

} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_EXAMPLE_H_
