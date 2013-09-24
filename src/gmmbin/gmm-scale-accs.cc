// gmmbin/gmm-scale-accs.cc

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

#include "util/common-utils.h"
#include "gmm/mle-am-diag-gmm.h"
#include "hmm/transition-model.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    typedef kaldi::int32 int32;

    const char *usage =
        "Scale GMM accumulators\n"
        "Usage: gmm-scale-accs [options] scale stats-in stats-out\n"
        "e.g.: gmm-scale-accs 0.5 1.stats half.stats\n";

    bool binary = true;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        scale_string = po.GetArg(1),
        stats_rxfilename = po.GetArg(2),
        stats_wxfilename = po.GetArg(3);

    BaseFloat scale;
    if (!ConvertStringToReal(scale_string, &scale))
      KALDI_ERR << "Invalid first argument to gmm-scale-accs: expect a number: "
                << scale_string;
          
    kaldi::Vector<double> transition_accs;
    kaldi::AccumAmDiagGmm gmm_accs;

    {
      bool binary_read;
      kaldi::Input ki(stats_rxfilename, &binary_read);
      transition_accs.Read(ki.Stream(), binary_read);
      gmm_accs.Read(ki.Stream(), binary_read);
    }
    transition_accs.Scale(scale);
    gmm_accs.Scale(scale);
    
    // Write out the scaled accs.
    {
      kaldi::Output ko(stats_wxfilename, binary);
      transition_accs.Write(ko.Stream(), binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Scaled accs with scale " << scale;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


