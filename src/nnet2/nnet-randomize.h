// nnet2/nnet-randomize.h

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

#ifndef KALDI_NNET2_NNET_RANDOMIZE_H_
#define KALDI_NNET2_NNET_RANDOMIZE_H_

#include "nnet2/nnet-update.h"
#include "nnet2/nnet-compute.h"
#include "itf/options-itf.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet2 {


/// Configuration variables that will be given to the program that
/// randomizes and weights the data for us.
struct NnetDataRandomizerConfig {
  int32 num_samples; // Total number of samples we want to train on (if >0).  The program
  // will select this many samples before it stops.

  BaseFloat num_epochs; // Total number of epochs we want (if >0).  The program will run
  // for this many epochs before it stops.

  NnetDataRandomizerConfig(): num_samples(-1), num_epochs(-1) { }

  void Register(OptionsItf *po) {
    po->Register("num-epochs", &num_epochs, "If >0, this will define how many "
                 "times to train on the whole data.  Note, you will see some "
                 "samples more than once if frequency-power < 1.0.  You must "
                 "define num-samples or num-epochs.");
    po->Register("num-samples", &num_samples, "The number of samples of training "
                 "data to train on.  You must define num-samples or num-epochs.");
  }

};

/// This class does the job of randomizing the data.
class NnetDataRandomizer {
 public:
  NnetDataRandomizer(int32 left_context,
                     int32 right_context,
                     const NnetDataRandomizerConfig &config);
      
  void AddTrainingFile(const Matrix<BaseFloat> &feats,
                       const Vector<BaseFloat> &spk_info,
                       const Posterior &pdf_post); // the "pdf_post" gives the
  // pdf-level posteriors, e.g. as output by post-to-pdf-post.
  
  bool Done();
  void Next();
  const NnetExample &Value();
  ~NnetDataRandomizer();
 private:
  void Init(); // This function is called the first time Next() or Value() is
  // called.
  
  /// Called from Next().
  void GetExample(const std::pair<int32, int32> &pair,
                  NnetExample *example) const;
  
  /// Called when samples_ is empty: sets up samples_.
  void RandomizeSamples(); 

  struct TrainingFile {
    CompressedMatrix feats;
    Vector<BaseFloat> spk_info;
    Posterior pdf_post; // pdf-level posteriors.  Typically a single
    // element per frame, with weight 1.0, for ML/Viterbi training.
    TrainingFile(const MatrixBase<BaseFloat> &feats_in,
                 const VectorBase<BaseFloat> &spk_info_in,
                 const Posterior &pdf_post_in):
        feats(feats_in), spk_info(spk_info_in), pdf_post(pdf_post_in) { }
  };
  
  int32 left_context_;
  int32 right_context_;
  NnetDataRandomizerConfig config_;    
  int32 num_samples_tgt_; // a function of the config.
  int32 num_samples_returned_; // increases during training.
  
  std::vector<TrainingFile*> data_;
  
  std::vector<std::pair<int32, int32> > samples_; // each time we randomize
  // the whole data, we store pairs here that record the (file, frame) index
  // of each randomized sample.  We pop elements off this list.
  
  NnetExample cur_example_; // Returned from Value().  NnetDataRandomizerConfig_ config_;
};



} // namespace nnet2
} // namespace kaldi

#endif

