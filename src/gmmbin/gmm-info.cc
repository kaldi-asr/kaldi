// gmmbin/gmm-info.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
        "Write to standard output various properties of GMM-based model\n"
        "Usage:  gmm-info [options] <model-in>\n"
        "e.g.:\n"
        " gmm-info 1.mdl\n"
        "See also: gmm-global-info, am-info\n";
    
    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    std::cout << "number of phones " << trans_model.GetPhones().size() << '\n';
    std::cout << "number of pdfs " << trans_model.NumPdfs() << '\n';
    std::cout << "number of transition-ids " << trans_model.NumTransitionIds()
              << '\n';
    std::cout << "number of transition-states "
              << trans_model.NumTransitionStates() << '\n';
    std::cout << "feature dimension " << am_gmm.Dim() << '\n';
    std::cout << "number of gaussians " << am_gmm.NumGauss() << '\n';
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


