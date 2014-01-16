// nnet2bin/nnet-combine-a.cc

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
#include "nnet2/combine-nnet-a.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This is a \"special case\" of neural net combination.  It takes a previous\n"
        "iteration's model, and N other models that have been trained in\n"
        "parallel with SGD on different batches.  If there are L updatable components,\n"
        "it first uses the validation set to train L parameters \\alpha_l, consisting of step-lengths\n"
        "along the direction (old-model) -> (average of trained models).  There is a threshold\n"
        "\"valid-impr-thresh\" (default 0.5).  If the validation-set improvement is more than\n"
        "this, we skip the next step.  The next step is to \"overshoot\" by a specified factor,\n"
        "e.g. 1.8.  (This should be strictly less than 2).  Once we have the resulting parameters\n"
        "\\alpha_l, we multiply the per-layer learning rates by those factors, subject to sanity-preserving\n"
        "limits on the changes and a minimum learning-rate (see the other options)\n"
        "\n"
        "Usage:  nnet-combine-a [options] <old-model> <new-model1> <new-model2> ... <new-modelN> <valid-examples-in> <model-out>\n"
        "\n"
        "e.g.:\n"
        " nnet-combine 1.1.nnet 1.2.nnet 1.3.nnet ark:valid.egs 2.nnet\n";
    
    bool binary_write = true;
    NnetCombineAconfig combine_config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    
    combine_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 4) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string
        nnet1_rxfilename = po.GetArg(1),
        valid_examples_rspecifier = po.GetArg(po.NumArgs() - 1),
        nnet_wxfilename = po.GetArg(po.NumArgs());
    
    TransitionModel trans_model;
    AmNnet am_nnet1;
    {
      bool binary_read;
      Input ki(nnet1_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet1.Read(ki.Stream(), binary_read);
    }

    int32 num_nnets = po.NumArgs() - 2;
    std::vector<Nnet> nnets(num_nnets);
    nnets[0] = am_nnet1.GetNnet();
    am_nnet1.GetNnet() = Nnet(); // Clear it to save memory.

    for (int32 n = 1; n < num_nnets; n++) {
      TransitionModel trans_model;
      AmNnet am_nnet;
      bool binary_read;
      Input ki(po.GetArg(1 + n), &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
      nnets[n] = am_nnet.GetNnet();
    }      
    
    std::vector<NnetExample> validation_set; // stores validation
    // frames.

    { // This block adds samples to "validation_set".
      SequentialNnetExampleReader example_reader(
          valid_examples_rspecifier);
      for (; !example_reader.Done(); example_reader.Next())
        validation_set.push_back(example_reader.Value());
      KALDI_LOG << "Read " << validation_set.size() << " examples from the "
                << "validation set.";
      KALDI_ASSERT(validation_set.size() > 0);
    }

        
    CombineNnetsA(combine_config,
                  validation_set,
                  nnets,
                  &(am_nnet1.GetNnet()));
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet1.Write(ko.Stream(), binary_write);
    }
    
    KALDI_LOG << "Finished combining neural nets, wrote model to "
              << nnet_wxfilename;
    return (validation_set.size() == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


