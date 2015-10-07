// ctcbin/nnet3-ctc-copy.cc

// Copyright 2015  Johns Hopkins University (author:  Daniel Povey)

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

#include <typeinfo>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "ctc/cctc-transition-model.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::ctc;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy nnet3+ctc acoustic model (which consists of CTC transition model\n"
        "followed by 'raw' nnet3 model), possibly changing binary mode.\n"
        "--raw=true option can be added if you want to just output the raw nnet.\n"
        "Also supports setting all learning rates to a supplied\n"
        "value (the --learning-rate option),\n"
        "and supports replacing the raw nnet in the model (the Nnet)\n"
        "with a provided raw nnet (the --set-raw-nnet option)\n"
        "Note: the --set-raw-nnet option can be used to initialize the model.\n"
        "\n"
        "Usage: nnet3-ctc-copy [options] <nnet3+ctc-model-in> <nnet3+ctc-model-out>\n"
        "e.g.:\n"
        " nnet3-ctc-copy --binary=false 0.mdl - | less\n"
        " nnet3-ctc-copy --raw=true 1.mdl 1.raw\n";

    bool binary_write = true,
        raw = false;
    BaseFloat learning_rate = -1;
    std::string set_raw_nnet = "";
    BaseFloat scale = 0.0;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("raw", &raw, "If true, write only the raw neural net, without "
                "the transition-model object");
    po.Register("learning-rate", &learning_rate,
                "Set all learning rates in the nnet to this value");
    po.Register("set-raw-nnet", &set_raw_nnet,
                "Use this option to set the raw nnet in the model to "
                "a provided neural net (provide an rxfilename)");
    po.Register("scale", &scale, "The parameter matrices are scaled"
                " by the specified value.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string ctc_nnet_rxfilename = po.GetArg(1),
        ctc_nnet_wxfilename = po.GetArg(2);

    CctcTransitionModel trans_model;
    Nnet nnet;
    {
      bool binary;
      Input input(ctc_nnet_rxfilename, &binary);
      trans_model.Read(input.Stream(), binary);
      if (set_raw_nnet.empty())
        nnet.Read(input.Stream(), binary);
    }

    if (!set_raw_nnet.empty())
      ReadKaldiObject(set_raw_nnet, &nnet);


    if (learning_rate >= 0)
      SetLearningRate(learning_rate, &nnet);

    if (scale != 0.0)
      ScaleNnet(scale, &nnet);

    if (raw) {
      WriteKaldiObject(nnet, ctc_nnet_wxfilename, binary_write);
      KALDI_LOG << "Copied nnet3+ctc neural net from " << ctc_nnet_rxfilename
                << " to raw format as " << ctc_nnet_wxfilename;

    } else {
      int32 nnet_output_dim = nnet.OutputDim("output"),
          cctc_output_dim = trans_model.NumOutputIndexes();
      if (nnet_output_dim != cctc_output_dim)
        KALDI_ERR << "Model output-dimension mismatch: nnet "
                  << nnet_output_dim << " vs. CTC model "
                  << cctc_output_dim;
      Output ko(ctc_nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      nnet.Write(ko.Stream(), binary_write);
      KALDI_LOG << "Copied nnet3+ctc neural net from " << ctc_nnet_rxfilename
                << " to " << ctc_nnet_wxfilename;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
