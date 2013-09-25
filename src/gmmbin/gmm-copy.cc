// gmmbin/gmm-copy.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy GMM based model (and possibly change binary/text format)\n"
        "Usage:  gmm-copy [options] <model-in> <model-out>\n"
        "e.g.:\n"
        " gmm-copy --binary=false 1.mdl 1_txt.mdl\n";


    bool binary_write = true,
        copy_am = true,
        copy_tm = true;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("copy-am", &copy_am, "Copy the acoustic model (AmDiagGmm object)");
    po.Register("copy-tm", &copy_tm, "Copy the transition model");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      if (copy_tm)
        trans_model.Read(ki.Stream(), binary_read);
      if (copy_am)
        am_gmm.Read(ki.Stream(), binary_read);
    }

    {
      Output ko(model_out_filename, binary_write);
      if (copy_tm)
        trans_model.Write(ko.Stream(), binary_write);
      if (copy_am)
        am_gmm.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


