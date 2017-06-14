// chainbin/nnet3-chain-combine.cc

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
#include "nnet3/nnet-chain-combine.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Using a subset of training or held-out nnet3+chain examples, compute an\n"
        "optimal combination of  anumber of nnet3 neural nets by maximizing the\n"
        "'chain' objective function.  See documentation of options for more details.\n"
        "Inputs and outputs are nnet3 raw nnets.\n"
        "\n"
        "Usage:  nnet3-chain-combine [options] <den-fst> <raw-nnet-in1> <raw-nnet-in2> ... <raw-nnet-inN> <chain-examples-in> <raw-nnet-out>\n"
        "\n"
        "e.g.:\n"
        " nnet3-combine den.fst 35.raw 36.raw 37.raw 38.raw ark:valid.cegs final.raw\n";

    bool binary_write = true;
    bool batchnorm_test_mode = false,
        dropout_test_mode = true;
    std::string use_gpu = "yes";
    NnetCombineConfig combine_config;
    chain::ChainTrainingOptions chain_config;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("batchnorm-test-mode", &batchnorm_test_mode,
                "If true, set test-mode to true on any BatchNormComponents.");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents.");

    combine_config.Register(&po);
    chain_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() < 4) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string
        den_fst_rxfilename = po.GetArg(1),
        raw_nnet_rxfilename = po.GetArg(2),
        valid_examples_rspecifier = po.GetArg(po.NumArgs() - 1),
        nnet_wxfilename = po.GetArg(po.NumArgs());


    fst::StdVectorFst den_fst;
    ReadFstKaldi(den_fst_rxfilename, &den_fst);

    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);

    if (batchnorm_test_mode)
      SetBatchnormTestMode(true, &nnet);
    if (dropout_test_mode)
      SetDropoutTestMode(true, &nnet);

    std::vector<NnetChainExample> egs;
    egs.reserve(10000);  // reserve a lot of space to minimize the chance of
                         // reallocation.

    { // This block adds training examples to "egs".
      SequentialNnetChainExampleReader example_reader(
          valid_examples_rspecifier);
      for (; !example_reader.Done(); example_reader.Next())
        egs.push_back(example_reader.Value());
      KALDI_LOG << "Read " << egs.size() << " examples.";
      KALDI_ASSERT(!egs.empty());
    }


    int32 num_nnets = po.NumArgs() - 3;
    NnetChainCombiner combiner(combine_config, chain_config,
                               num_nnets, egs, den_fst, nnet);

    for (int32 n = 1; n < num_nnets; n++) {
      std::string this_nnet_rxfilename = po.GetArg(n + 2);
      ReadKaldiObject(this_nnet_rxfilename, &nnet);
      combiner.AcceptNnet(nnet);
    }

    combiner.Combine();

    nnet = combiner.GetNnet();
    if (HasBatchnorm(nnet))
      RecomputeStats(egs, chain_config, den_fst, &nnet);

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    WriteKaldiObject(nnet, nnet_wxfilename, binary_write);

    KALDI_LOG << "Finished combining neural nets, wrote model to "
              << nnet_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
