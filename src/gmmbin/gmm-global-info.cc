// gmmbin/gmm-global-info.cc

// Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey)

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
#include "gmm/diag-gmm.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Write to standard output various properties of GMM model\n"
        "This is for a single diagonal GMM, e.g. as used for a UBM.\n"
        "Usage:  gmm-global-info [options] <gmm>\n"
        "e.g.:\n"
        " gmm-global-info 1.dubm\n"
        "See also: gmm-info, am-info\n";
    
    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1);

    DiagGmm gmm;
    ReadKaldiObject(model_rxfilename, &gmm);

    std::cout << "number of gaussians " << gmm.NumGauss() << '\n';
    std::cout << "feature dimension " << gmm.Dim() << '\n';
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


