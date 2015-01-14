// nnet2bin/nnet-combine-fast.cc

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
#include "nnet2/combine-nnet-fast.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Using a validation set, compute an optimal combination of a number of\n"
        "neural nets (the combination weights are separate for each layer and\n"
        "do not have to sum to one).  The optimization is BFGS, which is initialized\n"
        "from the best of the individual input neural nets (or as specified by\n"
        "--initial-model)\n"
        "\n"
        "Usage:  nnet-combine-fast [options] <model-in1> <model-in2> ... <model-inN> <valid-examples-in> <model-out>\n"
        "\n"
        "e.g.:\n"
        " nnet-combine-fast 1.1.nnet 1.2.nnet 1.3.nnet ark:valid.egs 2.nnet\n"
        "Caution: the first input neural net must not be a gradient.\n";
    
    bool binary_write = true;
    NnetCombineFastConfig combine_config;
    std::string use_gpu = "yes";
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    
    combine_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string
        nnet1_rxfilename = po.GetArg(1),
        valid_examples_rspecifier = po.GetArg(po.NumArgs() - 1),
        nnet_wxfilename = po.GetArg(po.NumArgs());

#if HAVE_CUDA==1
    if (combine_config.num_threads == 1)
      CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    
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

    CombineNnetsFast(combine_config,
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


