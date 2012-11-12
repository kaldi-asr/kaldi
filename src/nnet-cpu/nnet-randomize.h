// nnet-cpu/nnet-randomize.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET_CPU_NNET_RANDOMIZE_H_
#define KALDI_NNET_CPU_NNET_RANDOMIZE_H_

#include "nnet-cpu/nnet-update.h"
#include "nnet-cpu/nnet-compute.h"
#include "util/parse-options.h"

namespace kaldi {


/// Configuration variables that will be given to the program that
/// randomizes and weights the data for us.
struct NnetDataRandomizerConfig {
  /// If a particular class appears with a certain frequency f, we'll make it
  /// appear at frequency f^{frequency_power}, and reweight the samples with a
  /// higher weight to compensate.  Note: we'll give these weights an overall
  /// scale such that the expected weight of any given sample is 1; this helps
  /// keep this independent from the learning rates.  frequency_power=1.0
  /// means "normal" training.  We probably want 0.5 or so.
  BaseFloat frequency_power;
  
  int32 num_samples; // Total number of samples we want to train on (if >0).  The program
  // will select this many samples before it stops.

  BaseFloat num_epochs; // Total number of epochs we want (if >0).  The program will run
  // for this many epochs before it stops.

  NnetDataRandomizerConfig(): frequency_power(1.0), num_samples(-1),
                              num_epochs(-1) { }

  void Register(ParseOptions *po) {
    po->Register("frequency-power", &frequency_power, "Power by which we rescale "
                 "the frequencies of samples.");
    po->Register("num-epochs", &num_epochs, "If >0, this will define how many "
                 "times to train on the whole data.  Note, you will see some "
                 "samples more than once if frequency-power < 1.0.  You must "
                 "define num-samples or num-epochs.");
    po->Register("num-samples", &num_samples, "The number of samples of training "
                 "data to train on.  You must define num-samples or num-epochs.");
  }

};

/// This class does the job of randomizing and reweighting the data,
/// before training on it (the weights on samples are a mechanism
/// to make common classes less common, to avoid wasting time,
/// but then upweighting the samples so all the expectations are the
/// the same.
class NnetDataRandomizer {
 public:
  NnetDataRandomizer(int32 left_context,
                     int32 right_context,
                     const NnetDataRandomizerConfig &config);
      
  void AddTrainingFile(const Matrix<BaseFloat> &feats,
                       const Vector<BaseFloat> &spk_info,
                       const std::vector<int32> &labels);
  
  bool Done();
  void Next();
  const NnetTrainingExample &Value();
  ~NnetDataRandomizer();
 private:
  void Init(); // This function is called the first time Next() or Value() is
  // called.
  
  /// Called from RandomizeSamples().  Get samples indexed first
  /// by pdf-id, without any randomization or reweighting.
  void GetRawSamples(
      std::vector<std::vector<std::pair<int32, int32> > > *pdf_counts);
  /// Called from Next().
  void GetExample(const std::pair<int32, int32> &pair,
                  NnetTrainingExample *example) const;
  
  /// Called from RandomizeSamples().  Takes the samples indexed first by pdf,
  /// which are assumed to be in random order for each pdf, and writes them in
  /// pseudo-random order to *samples as one long sequence.  Uses a recursive
  /// algorithm (based on splitting in two) that is designed to ensure a kind
  /// of balance, e.g. each time we split in two we try to distribute examples
  /// of a pdf equally between the two splits.  This will tend to reduce
  /// the variance of the parameter estimates.  Note: the samples_by_pdf_input
  /// is the input but is destroyed by the algorithm to save memory.
  static void RandomizeSamplesRecurse(
      std::vector<std::vector<std::pair<int32, int32> > > *samples_by_pdf_input,
      std::vector<std::pair<int32, int32> > *samples);
  
  /// Called when samples_ is empty: sets
  /// up samples_ and pdf_weights_.  
  void RandomizeSamples(); 

  struct TrainingFile {
    CompressedMatrix feats;
    Vector<BaseFloat> spk_info;
    std::vector<int32> labels; // Vector of pdf-ids (targets for training).
    TrainingFile(const MatrixBase<BaseFloat> &feats_in,
                 const VectorBase<BaseFloat> &spk_info_in,
                 const std::vector<int32> &labels_in):
        feats(feats_in), spk_info(spk_info_in), labels(labels_in) { }
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

  Vector<BaseFloat> pdf_weights_; // each time we randomize the data,
  // we compute a new weighting for each pdf, which is to cancel out the
  // difference in frequency between the original frequency and the sampled
  // frequency.
  
  NnetTrainingExample cur_example_; // Returned from Value().  NnetDataRandomizerConfig_ config_;
};



} // namespace

#endif

