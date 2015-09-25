// ctcbin/nnet3-copy-transition-model.cc

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::ctc;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy CTC transition model (possibly changing binary mode)\n"
        "\n"
        "Usage:  ctc-copy-transition-model [options] <ctc-transition-model-in> <ctc-transition-model-out>\n"
        "e.g.:\n"
        " ctc-copy-transition-model --binary=false 0.trans_mdl - | less\n";

    bool binary_write = true;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string ctc_trans_model_rxfilename = po.GetArg(1),
                ctc_trans_model_wxfilename = po.GetArg(2);
    
    CctcTransitionModel trans_model;
    ReadKaldiObject(ctc_trans_model_rxfilename, &trans_model);
    
    WriteKaldiObject(trans_model, ctc_trans_model_wxfilename, binary_write);
    KALDI_LOG << "Copied CTC transition model from "
              << ctc_trans_model_rxfilename << " to "
              << ctc_trans_model_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
