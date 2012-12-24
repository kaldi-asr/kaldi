// nnet-dp/nnet-update-parallel.h

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

#ifndef KALDI_NNET_CPU_NNET_UPDATE_PARALLEL_H_
#define KALDI_NNET_CPU_NNET_UPDATE_PARALLEL_H_

#include "nnet-cpu/nnet-nnet.h"
#include "util/table-types.h"
#include "thread/kaldi-semaphore.h"
#include "nnet-cpu/nnet-update.h"

namespace kaldi {

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
/// All these examples will be treated as one minibatch.
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



} // namespace

#endif // KALDI_NNET_CPU_NNET_UPDATE_PARALLEL_H_
