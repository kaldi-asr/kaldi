// gmmbin/gmm-et-est-b.cc

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


int main(int argc, char *argv[])
{
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Update matrix B of exponential transform (uses stats from gmm-et-acc-b)\n"
        " [Use matrix-out with gmm-transform-means to transform model means.]\n"
        "Usage:  gmm-et-est-b [options] <et-in> <et-out> <matrix-out> <b-stats1> <b-stats2> ... \n"
        "e.g.: \n"
        " gmm-et-est-b 1.et 2.et 1.et_acc_b\n";

    bool binary = true;
    ParseOptions po(usage);

    std::string set_normalize_type = "";  // may be "", "none", "mean", or "mean-and-var";
    ExponentialTransformUpdateAOptions update_a_opts;
    po.Register("binary", &binary, "Write output in binary mode");
    update_a_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() < 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string et_rxfilename = po.GetArg(1);
    std::string et_wxfilename = po.GetArg(2);
    std::string mat_wxfilename = po.GetArg(3);

    ExponentialTransform et;
    {
      bool binary_in;
      Input ki(et_rxfilename, &binary_in);
      et.Read(ki.Stream(), binary_in);
    }
    ExponentialTransformAccsB stats;
    for (int32 i = 4; i <= po.NumArgs(); i++) {
      std::string stats_rxfilename = po.GetArg(i);
      bool binary_in;
      Input ki(stats_rxfilename, &binary_in);
      stats.Read(ki.Stream(), binary_in, true);  // true == add
    }

    int32 dim = et.Dim();
    Matrix<BaseFloat> M(dim, dim);  // to transform model means.
    stats.Update(&et, NULL, NULL, &M);

    {
      Output ko(et_wxfilename, binary);
      et.Write(ko.Stream(), binary);
    }
    {
      Output ko(mat_wxfilename, binary);
      M.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs and matrix.";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

