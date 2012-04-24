// gmmbin/gmm-et-est-a.cc

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
#include "transform/exponential-transform.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Update matrix A of exponential transform (uses stats from gmm-et-acc-a)\n"
        "Usage:  gmm-et-est-a [options] <et-in> <et-out> <a-stats1> <a-stats2> ... \n"
        "e.g.: \n"
        " gmm-et-est-a 1.et 2.et 1.et_acc_a\n";

    bool binary = true;
    ParseOptions po(usage);
    ExponentialTransformUpdateAOptions update_a_opts;
    po.Register("binary", &binary, "Write output in binary mode");
    update_a_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string et_rxfilename = po.GetArg(1);
    std::string et_wxfilename = po.GetArg(2);

    ExponentialTransform et;
    ReadKaldiObject(et_rxfilename, &et);
    ExponentialTransformAccsA stats;
    for (int32 i = 3; i <= po.NumArgs(); i++) {
      std::string stats_rxfilename = po.GetArg(i);
      bool binary_in;
      Input ki(stats_rxfilename, &binary_in);
      stats.Read(ki.Stream(), binary_in, true);  // true == add
    }

    stats.Update(update_a_opts, &et, NULL, NULL);

    WriteKaldiObject(et, et_wxfilename, binary);
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

