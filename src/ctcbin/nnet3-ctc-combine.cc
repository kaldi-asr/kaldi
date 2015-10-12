// ctcbin/nnet3-ctc-combine.cc

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
#include "nnet3/nnet-cctc-combine.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Using a subset of training or held-out nnet3+ctc examples, compute an\n"
        "optimal combination of  anumber of nnet3 neural nets by maximizing the\n"
        "CTC objective function.  See documentation of options for more details.\n"
        "Inputs and outputs are nnet3+ctc nnets.\n"
        "\n"
        "Usage:  nnet3-combine [options] <nnet-in1> <nnet-in2> ... <nnet-inN> <ctc-examples-in> <nnet-out>\n"
        "\n"
        "e.g.:\n"
        " nnet3-combine 35.mdl 36.mdl 37.mdl 38.mdl ark:valid.cegs final.mdl\n";

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


    ctc::CctcTransitionModel trans_mdl;
    Nnet nnet;
    {
      bool binary;
      Input input(nnet_rxfilename, &binary);
      trans_mdl.Read(input.Stream(), binary);
      nnet.Read(input.Stream(), binary);
    }


    std::vector<NnetCctcExample> egs;
    egs.reserve(10000);  // reserve a lot of space to minimize the chance of
                         // reallocation.

    { // This block adds training examples to "egs".
      SequentialNnetCctcExampleReader example_reader(
          valid_examples_rspecifier);
      for (; !example_reader.Done(); example_reader.Next())
        egs.push_back(example_reader.Value());
      KALDI_LOG << "Read " << egs.size() << " examples.";
      KALDI_ASSERT(!egs.empty());
    }


    int32 num_nnets = po.NumArgs() - 2;
    NnetCctcCombiner combiner(combine_config, num_nnets, egs, trans_mdl, nnet);


    for (int32 n = 1; n < num_nnets; n++) {
      ctc::CctcTransitionModel this_trans_mdl;
      bool binary;
      std::string this_nnet_rxfilename = po.GetArg(1 + n);
      Input input(this_nnet_rxfilename, &binary);
      this_trans_mdl.Read(input.Stream(), binary);
      if (!(this_trans_mdl == trans_mdl))
        KALDI_ERR << "Expected all transition-models to be identical.";
      nnet.Read(input.Stream(), binary);
      combiner.AcceptNnet(nnet);
    }

    combiner.Combine();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    {
      Output output(nnet_wxfilename, binary_write);
      trans_mdl.Write(output.Stream(), binary_write);
      combiner.GetNnet().Write(output.Stream(), binary_write);
    }
    KALDI_LOG << "Finished combining neural nets, wrote model to "
              << nnet_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


