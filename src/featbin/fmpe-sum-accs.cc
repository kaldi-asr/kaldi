// featbin/fmpe-sum-accs.cc

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
#include "transform/fmpe.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using kaldi::int32;
  try {
    const char *usage =
        "Sum fMPE stats\n"
        "Usage: fmpe-sum-accs [options...] <accs-out> <stats-in1> <stats-in2> ... \n"
        "E.g. fmpe-sum-accs 1.accs 1.1.accs 1.2.accs 1.3.accs 1.4.accs\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "If true, output fMPE stats in "
                "binary mode.");
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string stats_wxfilename = po.GetArg(1);

    FmpeStats stats;
    for (int32 arg = 2; arg <= po.NumArgs(); arg++) {
      std::string stats_rxfilename = po.GetArg(arg);
      bool binary;
      Input ki(stats_rxfilename, &binary);
      stats.Read(ki.Stream(), binary, true); // true == sum accs.
    }

    WriteKaldiObject(stats, stats_wxfilename, binary);
    
    KALDI_LOG << "Summed " << (po.NumArgs()-1) << " fMPE stats and wrote to "
              << stats_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
