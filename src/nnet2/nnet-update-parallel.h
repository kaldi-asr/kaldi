// nnet2/nnet-update-parallel.h

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

#ifndef KALDI_NNET2_NNET_UPDATE_PARALLEL_H_
#define KALDI_NNET2_NNET_UPDATE_PARALLEL_H_

#include "nnet2/nnet-nnet.h"
#include "util/table-types.h"
#include "thread/kaldi-semaphore.h"
#include "thread/kaldi-thread.h"
#include "itf/options-itf.h"
#include "nnet2/nnet-update.h"

namespace kaldi {
namespace nnet2 {

/** This struct stores neural net training examples to be used in
    multi-threaded training.  */
class ExamplesRepository {
 public:
  /// The following function is called by the code that reads in the examples,
  /// with a batch of examples.  [It will empty the vector "examples").
  void AcceptExamples(std::vector<NnetTrainingExample> *examples);

  /// The following function is called by the code that reads in the examples,
  /// when we're done reading examples.
  void ExamplesDone();
  
  /// This function is called by the code that does the training.  It gets the
  /// training examples, and if they are available, puts them in "examples" and
  /// returns true.  It returns false when there are no examples left and
  /// ExamplesDone() has been called.
  bool ProvideExamples(std::vector<NnetTrainingExample> *examples);
  
  ExamplesRepository(): empty_semaphore_(1), done_(false) { }
 private:
  Semaphore full_semaphore_;
  Semaphore empty_semaphore_;

  std::vector<NnetTrainingExample> examples_;
  bool done_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(ExamplesRepository);
};

/// This function is similar to "DoBackprop" in nnet-update.h
/// This function computes the objective function and either updates the model
/// or computes parameter gradients.  It returns the cross-entropy objective
/// function summed over all samples (normalize this by
/// TotalNnetTrainingWeight(examples)).  It is mostly a wrapper for
/// a class NnetUpdater that's defined in nnet-update.cc, but we
/// don't want to expose that complexity at this level.
/// Note: this function 
/// If &nnet == nnet_to_update, it assumes we're doing SGD and does
/// something like Hogwild; otherwise it assumes we're computing a
/// gradient and it sums up the gradients.
/// The return value is the total log-prob summed over the #frames. It also
/// outputs the #frames into "num_frames".
BaseFloat DoBackpropParallel(const Nnet &nnet,
                             int32 minibatch_size,
                             SequentialNnetTrainingExampleReader *example_reader,
                             int64 *num_frames,
                             Nnet *nnet_to_update);

struct SafeBackpropConfig {
  BaseFloat max_degradation; // Max degradation per test sample that is
                                 // allowed over the process's duration.
  int32 num_samples; // The approximate number of examples that we expect to train on.  (Since
                     // we read from archives, this is not known upfront).  This should match
                     // the number of examples in the input.   This will normally be several
                     // hundred thousand, but we don't set a default.

  int32 samples_per_test; // Number of samples to use to compute the objective function
                          // derivatives, each time.

  BaseFloat relax_factor;
  
  SafeBackpropConfig(): max_degradation(1.0), num_samples(-1),
                        samples_per_test(2000), relax_factor(1.1) { }
  
  void Register(OptionsItf *po) {
    po->Register("max-degradation", &max_degradation, "Maximum degradation "
                 "allowed over the process lifetime, in safe-backprop.  You must "
                 "set --num-samples too.");
    po->Register("num-samples", &num_samples, "The number of samples you intend "
                 "to train on-- if set, activates safe-backprop code.  Must match "
                 "actual number of samples you will train on in this process.");
    po->Register("samples-per-test", &samples_per_test, "In safe-backprop code, "
                 "the number of samples to use each time we consider changing "
                 "the learning rates.");
    po->Register("relax-factor", &relax_factor, "In safe-backprop code, "
                 "factor by which we allow learning rates to relax to their "
                 "original values, each time.");
  }
  
};


/// DoBackpropParallelSafe is as DoBackpropParallel, but takes an extra
/// configuration class "SafeBackpropConfig".  This specifies
/// per-frame objective function is allowed to degrade per training example,(his
/// will be equal to a reasonable per-frame degradation, such as 1.0, divided
/// by the #samples in the training run, e.g. 200k).  
/// This needs a little explanation.  When we run parallel SGD runs on different
/// machines and then average the parameters at the end of each iteration, what
/// we typically find is that for each individual job the objective function
/// (whether measured on test data or validation data) is worse than at the start,
/// but after averaging it's better.  This is because of reduced parameter noise
/// after averaging.  However, if the individual jobs get *too much* worse
/// (e.g. the log-prob per frame is worse by more than about 1.0), we can't
/// always recover via averaging.  In this function we attempt to put a limit on
/// how much degradation we are willing to accept.  Rather than an absolute
/// limit we impose a limit per training sample so that, say, halfway through
/// training the amount of degradation we accept is half what we'd accept at the
/// end.  This will make the learning rates more consistent throughout the
/// training run.  This function basically applies the limit by controlling the
/// learning rates, but if at a certain point it decreases the learning rate of
/// a layer by a certain factor, it also scales the changes in parameters of
/// that layer by the same factor, as if it had "retroactively applied" the
/// learning rate change.
BaseFloat DoBackpropParallelSafe(
    int32 minibatch_size,
    const SafeBackpropConfig &safe_config,
    SequentialNnetTrainingExampleReader *example_reader,
    int64 *num_frames,
    Nnet *nnet);

/// This function is like DoBackpropParallel() but incorporates momentum.  We
/// formulate momentum in such a way that it doesn't change the effective
/// learning rate.  momentum_minibatches is the number of minibatches that is
/// the time-constant for the momentum, so we update the model gradually over
/// "momentum_minibatches" minibatches.  This number of minibatches is the
/// global number over all threads, not per thread.  Caution: we effectively
/// lose a little data at the end of the run, corresponding to approximately
/// "momentum_minibatches" batches of data.
BaseFloat DoBackpropParallelMomentum(
    int32 minibatch_size,
    BaseFloat momentum_minibatches,
    SequentialNnetTrainingExampleReader *example_reader,
    int64 *num_frames,
    Nnet *nnet);

/// This version of DoBackpropParallel takes a vector of examples, and will
/// typically be used to compute the exact gradient. 
BaseFloat DoBackpropParallel(const Nnet &nnet,
                             int32 minibatch_size,
                             int32 num_threads,
                             const std::vector<NnetTrainingExample> &examples,
                             int64 *num_frames,
                             Nnet *nnet_to_update);



} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET2_NNET_UPDATE_PARALLEL_H_
