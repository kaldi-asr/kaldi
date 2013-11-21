// nnet2bin/nnet-get-preconditioner.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet2/nnet-randomize.h"
#include "nnet2/nnet-lbfgs.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Create a neural-net model that can be used as a preconditioner\n"
        "by programs like nnet-combine (contains a component of type AffineComponentA\n"
        "corresponding to each descendant of AffineComponent in the original net.)\n"
        "\n"
        "Usage:  nnet-get-preconditioner [options] <model-in> <training-examples-in> <preconditioner-out>\n"
        "\n"
        "e.g.:\n"
        "nnet-get-preconditioner 1.nnet ark:1.egs 1.preconditioner\n";
    

    int32 minibatch_size = 1024;
    bool binary_write = true;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("minibatch-size", &minibatch_size,
                "Size of minibatches used in computation");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        preconditioner_wxfilename = po.GetArg(3);


    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    Nnet *preconditioner = GetPreconditioner(am_nnet.GetNnet());    
    AmNnet am_preconditioner(*preconditioner);
    delete preconditioner;
    preconditioner = NULL;
    
    int64 num_examples = 0;
    double tot_logprob = 0;
    
    std::vector<NnetExample> examples;
    
    SequentialNnetExampleReader example_reader(examples_rspecifier);
    for (; !example_reader.Done(); example_reader.Next()) {
      examples.push_back(example_reader.Value());
      num_examples++;
      if (static_cast<int32>(examples.size()) == minibatch_size) {
        tot_logprob += DoBackprop(am_nnet.GetNnet(),
                                  examples,
                                  &(am_preconditioner.GetNnet()));
        examples.clear();
      }
      if (num_examples % 100000 == 0 && num_examples > 0)
        KALDI_LOG << "Processed " << (num_examples - examples.size())
                  << " examples, average log-prob per example is "
                  << (tot_logprob / (num_examples - examples.size()));
    }
    if (!examples.empty())
      tot_logprob += DoBackprop(am_nnet.GetNnet(),
                                examples,
                                &(am_preconditioner.GetNnet()));
    
    { // Write the preconditioner.
      Output ko(preconditioner_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_preconditioner.Write(ko.Stream(), binary_write);
    }
    
    KALDI_LOG << "Overall log-prob per example was "
              << (tot_logprob / num_examples) << " over "
              << num_examples << " examples.";
    KALDI_LOG << "Wrote preconditioner to "
              << preconditioner_wxfilename;
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


