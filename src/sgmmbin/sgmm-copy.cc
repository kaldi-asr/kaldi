// sgmmbin/sgmm-copy.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Copy SGMM (possibly changing binary/text format)\n"
        "Usage: sgmm-normalize [options] <model-in> <model-out>\n"
        "e.g.: sgmm-normalize --binary=false 1.mdl 1_text.mdl\n";

    bool binary_write = true;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    AmSgmm am_sgmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input is(model_in_filename, &binary);
      trans_model.Read(is.Stream(), binary);
      am_sgmm.Read(is.Stream(), binary);
    }

    {
      Output os(model_out_filename, binary_write);
      trans_model.Write(os.Stream(), binary_write);
      am_sgmm.Write(os.Stream(), binary_write, kSgmmWriteAll);
    }
    
    
    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


