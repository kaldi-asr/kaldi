// nnet3bin/nnet3-combine.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet3/nnet-combine.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Using a subset of training or held-out examples, compute an optimal combination of a\n"
        "number of nnet3 neural nets by maximizing the objective function.  See documentation of\n"
        "options for more details.  Inputs and outputs are 'raw' nnets.\n"
        "\n"
        "Usage:  nnet3-combine [options] <nnet-in1> <nnet-in2> ... <nnet-inN> <valid-examples-in> <nnet-out>\n"
        "\n"
        "e.g.:\n"
        " nnet3-combine 1.1.raw 1.2.raw 1.3.raw ark:valid.egs 2.raw\n";
    
    bool binary_write = true;
    std::string use_gpu = "yes";    
    NnetCombineConfig combine_config;
    
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

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
    
    std::string
        nnet_rxfilename = po.GetArg(1),
        valid_examples_rspecifier = po.GetArg(po.NumArgs() - 1),
        nnet_wxfilename = po.GetArg(po.NumArgs());

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);
    

    std::vector<NnetExample> egs;
    egs.reserve(10000);  // reserve a lot of space to minimize the chance of
                         // reallocation.

    { // This block adds training examples to "egs".
      SequentialNnetExampleReader example_reader(
          valid_examples_rspecifier);
      for (; !example_reader.Done(); example_reader.Next())
        egs.push_back(example_reader.Value());
      KALDI_LOG << "Read " << egs.size() << " examples.";
      KALDI_ASSERT(!egs.empty());
    }
    
    
    int32 num_nnets = po.NumArgs() - 2;
    if (num_nnets > 1 || !combine_config.enforce_sum_to_one) {
      NnetCombiner combiner(combine_config, num_nnets, egs, nnet);
      
      for (int32 n = 1; n < num_nnets; n++) {
        ReadKaldiObject(po.GetArg(1 + n), &nnet);
        combiner.AcceptNnet(nnet);
      }

      combiner.Combine();


#if HAVE_CUDA==1
      CuDevice::Instantiate().PrintProfile();
#endif

      WriteKaldiObject(combiner.GetNnet(), nnet_wxfilename, binary_write);
    } else {
      KALDI_LOG << "Copying the single input model directly to the output, "
                << "without any combination.";
      SetDropoutProportion(0, &nnet);
      WriteKaldiObject(nnet, nnet_wxfilename, binary_write);
    } 
    KALDI_LOG << "Finished combining neural nets, wrote model to "
              << nnet_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


