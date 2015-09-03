// featbin/fmpe-est.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)  Yanmin Qian

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
  try {
    const char *usage =
        "Do one iteration of learning (modified gradient descent)\n"
        "on fMPE transform\n"
        "Usage: fmpe-est [options...] <fmpe-in> <stats-in> <fmpe-out>\n"
        "E.g. fmpe-est 1.fmpe 1.accs 2.fmpe\n";

    ParseOptions po(usage);
    FmpeUpdateOptions opts;
    bool binary = true;
    po.Register("binary", &binary, "If true, output fMPE object in "
                "binary mode.");
    opts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string fmpe_rxfilename = po.GetArg(1),
        stats_rxfilename = po.GetArg(2),
        fmpe_wxfilename = po.GetArg(3);

    Fmpe fmpe;
    ReadKaldiObject(fmpe_rxfilename, &fmpe);
    FmpeStats stats;
    ReadKaldiObject(stats_rxfilename, &stats);

    stats.DoChecks(); // checks certain checksums.
    fmpe.Update(opts, stats);

    WriteKaldiObject(fmpe, fmpe_wxfilename, binary);

    KALDI_LOG << "Updated fMPE object and wrote to "
              << fmpe_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
