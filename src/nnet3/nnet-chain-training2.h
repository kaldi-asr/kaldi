// nnet3/nnet-chain-training.h

// Copyright    2015  Johns Hopkins University (author: Daniel Povey)
//              2019  Idiap Research Institute (author: Srikanth Madikeri)

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

#ifndef KALDI_NNET3_NNET_CHAIN_TRAINING2_H_
#define KALDI_NNET3_NNET_CHAIN_TRAINING2_H_

#include "chain/chain-den-graph.h"
#include "chain/chain-training.h"
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-chain-training.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-training.h"

namespace kaldi {
namespace nnet3 {

struct NnetChainTraining2Options {
  NnetTrainerOptions nnet_config;
  chain::ChainTrainingOptions chain_config;
  bool apply_deriv_weights;
  NnetChainTraining2Options(): apply_deriv_weights(true) { }

  void Register(OptionsItf *opts) {
    nnet_config.Register(opts);
    chain_config.Register(opts);
    opts->Register("apply-deriv-weights", &apply_deriv_weights,
                   "If true, apply the per-frame derivative weights stored "
                   "with the example");
  }
};

class NnetChainModel2 {
 public:
  /**
     Constructor to which you pass the model and the den-fst
     directory.  There is no requirement that all these directories be distinct.

     For each language called "lang" the following files should exist:
       <den_fst_dir>/lang.den.fst <den_fst_dir>/lang.normalization.fst

     In practice, the language name will be either "default", in the
     typical (monolingual) setup, or it might be arbitrary strings
     representing languages such as "english", "french", and so on.
     In general the language can be any string containing ASCII letters, numbers
     or underscores.

     The models and denominator FSTs will only be read when they are actually
     required, so languages that are not used by a particular job (e.g. because
     they were not represented in the egs) will not actually be read.

      **/

  NnetChainModel2(Nnet *nnet, const std::string &den_fst_dir)
      : nnet_(nnet), den_fst_dir_(den_fst_dir) {}

  const chain::DenominatorGraph *GetDenGraphForLang(const std::string &lang);

 private:
  // struct LanguageInfo contains the data that is stored per language.
  // transform comes from <transform_dir>/<language_name>.ada
  struct LanguageInfo {
    LanguageInfo() = delete;
    LanguageInfo(const LanguageInfo&) = default;
    LanguageInfo(const std::string &name,
                 const fst::StdVectorFst &den_fst,
                 int32 num_pdfs)
        : name(name), den_graph(den_fst, num_pdfs) {}

    /// Language name.
    std::string name;
    /// Denominator loaded from '<language_name>.den.fst'.
    chain::DenominatorGraph den_graph;
  };

  // Get the LanguageInfo* for this language, creating it (and reading its
  // contents from disk) if it does not already exist.
  const LanguageInfo *GetInfoForLang(const std::string &lang);

  Nnet *nnet_;
  // Directory where denominator FSTs are located.
  std::string den_fst_dir_;

  std::unordered_map<std::string, LanguageInfo, StringHasher> lang_info_;
};


/**
   This class is for single-threaded training of neural nets using the 'chain'
   model.
*/
class NnetChainTrainer2 {
 public:
  NnetChainTrainer2(const NnetChainTraining2Options &config,
                    const NnetChainModel2 &model,
                    Nnet *nnet);

  ~NnetChainTrainer2();

  // train on one minibatch.
  void Train(const std::string &key, NnetChainExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

 private:
  // The internal function for doing one step of conventional SGD training.
  void TrainInternal(const std::string &key, const NnetChainExample &eg,
                     const NnetComputation &computation,
                     const std::string &lang_name);

  // The internal function for doing one step of backstitch training. Depending
  // on whether is_backstitch_step1 is true, It could be either the first
  // (backward) step, or the second (forward) step of backstitch.
  void TrainInternalBackstitch(const std::string key,
                               const NnetChainExample &eg,
                               const NnetComputation &computation,
                               bool is_backstitch_step1);

  void ProcessOutputs(bool is_backstitch_step2, const std::string &key,
                      const NnetChainExample &eg, NnetComputer *computer);

  const NnetChainTraining2Options opts_;

  NnetChainModel2 model_;
  Nnet *nnet_;
  Nnet *delta_nnet_;  // stores the change to the parameters on each training
                      // iteration.
  CachingOptimizingCompiler compiler_;

  // This code supports multiple output layers, even though in the
  // normal case there will be just one output layer named "output".
  // So we store the objective functions per output layer.
  int32 num_minibatches_processed_;

  // stats for max-change.
  MaxChangeStats max_change_stats_;

  unordered_map<std::string, ObjectiveFunctionInfo, StringHasher> objf_info_;

  // This value is used in backstitch training when we need to ensure
  // consistent dropout masks.  It's set to a value derived from rand()
  // when the class is initialized.
  int32 srand_seed_;
};


}  // namespace nnet3
}  // namespace kaldi

#endif // KALDI_NNET3_NNET_CHAIN_TRAINING2_H_
