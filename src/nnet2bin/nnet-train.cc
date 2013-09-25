// nnet2bin/nnet-train.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet2/nnet-randomize.h"
#include "nnet2/train-nnet.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train the neural network parameters with backprop and stochastic\n"
        "gradient descent using minibatches.  The training frames and labels\n"
        "are read via a pipe from nnet-randomize-frames.  This program uses a\n"
        "heuristic based on the validation set gradient to periodically update\n"
        "the learning rate of each layer.  It will train until the data from\n"
        "nnet-randomize-frames finishes. The validation-set training data is\n"
        "directly input to this program.\n"
        "\n"
        "Usage:  nnet-train [options] <model-in> <training-examples-in> <valid-examples-in> <model-out>\n"
        "\n"
        "e.g.:\n"
        "nnet-randomize-frames [args] | nnet-train 1.nnet ark:- ark:valid.egs 2.nnet\n";
    
    bool binary_write = true;
    bool zero_stats = true;
    int32 srand_seed = 0;
    NnetAdaptiveTrainerConfig train_config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    // TODO: remove next statement (old name).
    po.Register("zero-occupancy", &zero_stats, "If true, zero occupation "
                "counts stored with the neural net (only affects mixing up).");
    po.Register("zero-stats", &zero_stats, "If true, zero occupation "
                "counts stored with the neural net (only affects mixing up).");
    po.Register("srand", &srand_seed,
                "Seed for random number generator (e.g., for dropout)");
    
    train_config.Register(&po);
    
    po.Read(argc, argv);
    srand(srand_seed);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        valid_examples_rspecifier = po.GetArg(3),
        nnet_wxfilename = po.GetArg(4);


    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    if (zero_stats)
      am_nnet.GetNnet().ZeroStats();
    
    std::vector<NnetTrainingExample> validation_set; // stores validation
    // frames.

    { // This block adds samples to "validation_set".
      SequentialNnetTrainingExampleReader example_reader(
          valid_examples_rspecifier);
      for (; !example_reader.Done(); example_reader.Next())
        validation_set.push_back(example_reader.Value());
      KALDI_LOG << "Read " << validation_set.size() << " examples from the "
                << "validation set.";
      KALDI_ASSERT(validation_set.size() > 0);
    }

    int64 num_examples = 0;
    { // want to make sure this object deinitializes before
      // we write the model, as it does something in the destructor.
      NnetAdaptiveTrainer trainer(train_config,
                                  validation_set,
                                  &(am_nnet.GetNnet()));
    
      SequentialNnetTrainingExampleReader example_reader(examples_rspecifier);

      for (; !example_reader.Done(); example_reader.Next(), num_examples++)
        trainer.TrainOnExample(example_reader.Value());  // It all happens here!
    }
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    
    KALDI_LOG << "Finished training, processed " << num_examples
              << " training examples.  Wrote model to "
              << nnet_wxfilename;
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


