// nnet-cpu/train-nnet.h

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

#ifndef KALDI_NNET_CPU_TRAIN_NNET_H_
#define KALDI_NNET_CPU_TRAIN_NNET_H_

#include "nnet-cpu/nnet-update.h"
#include "nnet-cpu/nnet-compute.h"
#include "util/parse-options.h"

namespace kaldi {



/// Class NnetValidationSet stores the validation set feature data and labels,
/// and is responsible for calling code that computes the objective function and
/// gradient on the validation set.
class NnetValidationSet {
 public:
  NnetValidationSet() { }

  /// This is used while initializing the object.
  void AddUtterance(const MatrixBase<BaseFloat> &features,
                    const VectorBase<BaseFloat> &spk_info, // may be empty
                    const std::vector<int32> &pdf_ids,
                    BaseFloat utterance_weight = 1.0);

  /// Here, "nnet" will be a neural net and "gradient" will be a copy of it that
  /// this function will overwrite with the gradient.  This function will compute
  /// the gradient and return the *average* per-frame objective function
  BaseFloat ComputeGradient(const Nnet &nnet,
                            Nnet *gradient) const;

  /// Returns the total #frames weighted by the utterance_weight (which is
  /// typically one).
  BaseFloat TotalWeight() const; 
                    
  ~NnetValidationSet();
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(NnetValidationSet);
  struct Utterance {
    Matrix<BaseFloat> features;
    Vector<BaseFloat> spk_info;
    std::vector<int32> pdf_ids;
    BaseFloat weight;
    Utterance(const MatrixBase<BaseFloat> &features_in,
              const VectorBase<BaseFloat> &spk_info_in,
              const std::vector<int32> &pdf_ids_in,
              BaseFloat weight_in): features(features_in),
                                    spk_info(spk_info_in),
                                    pdf_ids(pdf_ids_in),
                                    weight(weight_in) { }
  };
  std::vector<Utterance*> utterances_;
};
  

struct NnetAdaptiveTrainerConfig {
  int32 minibatch_size;
  int32 minibatches_per_phase;
  BaseFloat learning_rate_ratio;
  BaseFloat measure_gradient_at;
  BaseFloat max_learning_rate;
  BaseFloat min_l2_penalty;
  BaseFloat max_l2_penalty;
  int32 num_phases;
  
  NnetAdaptiveTrainerConfig():
      minibatch_size(500), minibatches_per_phase(50),
      learning_rate_ratio(1.1),
      measure_gradient_at(1.0),
      max_learning_rate(0.1),
      min_l2_penalty(1.0e-10), max_l2_penalty(1.0) { }
  
  void Register (ParseOptions *po) {
    po->Register("minibatch-size", &minibatch_size,
                 "Number of samples per minibatch of training data.");
    po->Register("minibatches-per-phase", &minibatches_per_phase,
                 "Number of minibatches accessed in each phase of training "
                 "(after each phase we adjust learning rates");
    po->Register("learning-rate-ratio", &learning_rate_ratio,
                 "Ratio by which we change the learning and shrinkage rates "
                 "in each phase of training (can get larger or smaller by "
                 "this factor).");
    po->Register("max-learning-rate", &max_learning_rate,
                 "Maximum learning rate we allow when dynamically updating "
                 "learning and shrinkage rates");
    po->Register("min-l2-penalty", &min_l2_penalty,
                 "Minimum allowed l2 penalty.");
    po->Register("max-l2-penalty", &max_l2_penalty,
                 "Maximum allowed l2 penalty.");
    po->Register("measure-gradient-at", &measure_gradient_at,
                 "This parameter controls automatic setting of learning rates. "
                 "Must be >0.5 and >= 1.0.  Closer to 0.5 leads to faster "
                 "learning rates.");
  }  
};

// Class NnetAdaptiveTrainer is responsible for changing the learning rate using
// the gradients on the validation set, and calling the SGD training code in
// nnet-update.h.  It takes in the training examples through the call
// "TrainOnExample()", which means that the I/O code that reads in the training
// examples can be in the .cc file (we prefer to segregate that out).
class NnetAdaptiveTrainer {
 public:
  NnetAdaptiveTrainer(const NnetAdaptiveTrainerConfig &config,
                      const std::vector<NnetTrainingExample> &validation_set,
                      Nnet *nnet);
  
  /// TrainOnExample will take the example and add it to a buffer;
  /// if we've reached the minibatch size it will do the training.
  void TrainOnExample(const NnetTrainingExample &value);

  ~NnetAdaptiveTrainer();
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(NnetAdaptiveTrainer);

  // returns objf per sample, sets "gradient" to the gradient
  // on the validation set, computed at value *nnet_.
  BaseFloat ComputeValidationSetGradient(Nnet *gradient) const;
  
  void TrainOneMinibatch();
  
  // The following function is called by TrainOneMinibatch()
  // when we enter a new phase.
  void BeginNewPhase(bool first_time);
  
  // Things we were given in the initializer:
  NnetAdaptiveTrainerConfig config_;
  const std::vector<NnetTrainingExample> &validation_set_; // Stores validation data, used
  BaseFloat validation_tot_weight_;
  // to compute gradient on validation set.
  Nnet *nnet_; // the nnet we're training.

  // State information:
  int32 num_phases_;
  int32 minibatches_seen_this_phase_;
  std::vector<NnetTrainingExample> buffer_;
  BaseFloat validation_objf_; // stores validation objective function at
  // start/end of phase.
  Nnet validation_gradient_; // validation gradient at start of this phase.
  Nnet nnet_snapshot_; // snapshot of nnet params at start of this phase.
  double logprob_this_phase_; // Needed for accumulating train log-prob on each phase.
  double weight_this_phase_; // weight corresponding to the above.
  
  // Stuff that's not really specific to a phase:
  BaseFloat initial_validation_objf_; // validation objf at start.
  Vector<BaseFloat> progress_stats_; // Per-layer stats on progress so far.
};



struct NnetSimpleTrainerConfig {
  int32 minibatch_size;
  int32 minibatches_per_phase;
  BaseFloat learning_rate_ratio;
  BaseFloat measure_gradient_at;
  BaseFloat max_learning_rate;
  BaseFloat min_l2_penalty;
  BaseFloat max_l2_penalty;
  int32 num_phases;
  
  NnetSimpleTrainerConfig():
      minibatch_size(500), minibatches_per_phase(50),
      learning_rate_ratio(1.1),
      measure_gradient_at(1.0),
      max_learning_rate(0.1),
      min_l2_penalty(1.0e-10), max_l2_penalty(1.0) { }
  
  void Register (ParseOptions *po) {
    po->Register("minibatch-size", &minibatch_size,
                 "Number of samples per minibatch of training data.");
    po->Register("minibatches-per-phase", &minibatches_per_phase,
                 "Number of minibatches accessed in each phase of training "
                 "(after each phase we adjust learning rates");
    po->Register("learning-rate-ratio", &learning_rate_ratio,
                 "Ratio by which we change the learning and shrinkage rates "
                 "in each phase of training (can get larger or smaller by "
                 "this factor).");
    po->Register("max-learning-rate", &max_learning_rate,
                 "Maximum learning rate we allow when dynamically updating "
                 "learning and shrinkage rates");
    po->Register("min-l2-penalty", &min_l2_penalty,
                 "Minimum allowed l2 penalty.");
    po->Register("max-l2-penalty", &max_l2_penalty,
                 "Maximum allowed l2 penalty.");
    po->Register("measure-gradient-at", &measure_gradient_at,
                 "This parameter controls automatic setting of learning rates. "
                 "Must be >0.5 and >= 1.0.  Closer to 0.5 leads to faster "
                 "learning rates.");
  }  
};

class NnetSimpleTrainer {
 public:
  NnetSimpleTrainer(const NnetSimpleTrainerConfig &config,
                    const std::vector<NnetTrainingExample> &validation_set,
                    Nnet *nnet);
  
  /// TrainOnExample will take the example and add it to a buffer;
  /// if we've reached the minibatch size it will do the training.
  void TrainOnExample(const NnetTrainingExample &value);

  ~NnetSimpleTrainer();
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(NnetSimpleTrainer);

  // returns objf per sample, sets "gradient" to the gradient
  // on the validation set, computed at value *nnet_.
  BaseFloat ComputeValidationSetGradient(Nnet *gradient) const;
  
  void TrainOneMinibatch();
  
  // The following function is called by TrainOneMinibatch()
  // when we enter a new phase.
  void BeginNewPhase(bool first_time);
  
  // Things we were given in the initializer:
  NnetSimpleTrainerConfig config_;
  const std::vector<NnetTrainingExample> &validation_set_; // Stores validation data, used
  BaseFloat validation_tot_weight_;
  // to compute gradient on validation set.
  Nnet *nnet_; // the nnet we're training.

  // State information:
  int32 num_phases_;
  int32 minibatches_seen_this_phase_;
  std::vector<NnetTrainingExample> buffer_;
  BaseFloat validation_objf_; // stores validation objective function at
  // start/end of phase.
  Nnet validation_gradient_; // validation gradient at start of this phase.
  Nnet nnet_snapshot_; // snapshot of nnet params at start of this phase.
  double logprob_this_phase_; // Needed for accumulating train log-prob on each phase.
  double weight_this_phase_; // weight corresponding to the above.
  
  // Stuff that's not really specific to a phase:
  BaseFloat initial_validation_objf_; // validation objf at start.
  Vector<BaseFloat> progress_stats_; // Per-layer stats on progress so far.
};



} // namespace

#endif
