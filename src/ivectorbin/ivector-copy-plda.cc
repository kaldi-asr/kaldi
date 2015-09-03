// ivectorbin/ivector-copy-plda.cc

// Copyright 2013  Daniel Povey

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
#include "ivector/plda.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Copy a PLDA object, possibly applying smoothing to the within-class\n"
        "covariance\n"
        "\n"
        "Usage: ivector-copy-plda <plda-in> <plda-out>\n"
        "e.g.: ivector-copy-plda --smoothing=0.1 plda plda.smooth0.1\n";
    
    ParseOptions po(usage);

    BaseFloat smoothing = 0.0;
    bool binary = true;
    po.Register("smoothing", &smoothing, "Factor used in smoothing within-class "
                "covariance (add this factor times between-class covar)");
    po.Register("binary", &binary, "Write output in binary mode");
    
    PldaConfig plda_config;
    plda_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_rxfilename = po.GetArg(1),
        plda_wxfilename = po.GetArg(2);

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);
    if (smoothing != 0.0)
      plda.SmoothWithinClassCovariance(smoothing);
    WriteKaldiObject(plda, plda_wxfilename, binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
