// fgmmbin/fgmm-global-to-gmm.cc

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

#include "util/common-utils.h"
#include "gmm/full-gmm.h"
#include "gmm/mle-full-gmm.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Convert single full-covariance GMM to single diagonal-covariance GMM.\n"
        "Usage: fgmm-global-to-gmm [options] 1.fgmm 1.gmm\n";
        
    bool binary = true;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fgmm_rxfilename = po.GetArg(1),
        gmm_wxfilename = po.GetArg(2);
    
    FullGmm fgmm;
    
    {
      bool binary_read;
      Input ki(fgmm_rxfilename, &binary_read);
      fgmm.Read(ki.Stream(), binary_read);
    }

    DiagGmm gmm;
    gmm.CopyFromFullGmm(fgmm);
    WriteKaldiObject(gmm, gmm_wxfilename, binary);
    KALDI_LOG << "Written diagonal GMM to " << gmm_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

