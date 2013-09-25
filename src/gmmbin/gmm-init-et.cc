// gmmbin/gmm-init-et.cc

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
#include "transform/exponential-transform.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialize exponential tranform\n"
        "Usage:  gmm-init-et [options] <et-object-out>\n"
        "e.g.: \n"
        " gmm-init-et --dim=39  1.et\n";

    bool binary = true;
    int32 dim = 13;
    int32 seed = 0;
    std::string normalize_type = "offset";

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("dim", &dim, "Feature dimension");
    po.Register("seed", &seed, "Seed for random initialization of A matrix.");
    po.Register("normalize-type", &normalize_type, "Normalization type: \"offset\"|\"diag\"|\"none\"");

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    EtNormalizeType norm_type = kEtNormalizeNone;
    if (normalize_type == "offset") norm_type = kEtNormalizeOffset;
    else if (normalize_type == "diag") norm_type = kEtNormalizeDiag;
    else if (normalize_type == "none") norm_type = kEtNormalizeNone;
    else
      KALDI_ERR << "Invalid option --normalize-type=" << normalize_type;

    std::string et_wxfilename = po.GetArg(1);

    ExponentialTransform et(dim, norm_type, seed);
    Output ko(et_wxfilename, binary);
    et.Write(ko.Stream(), binary);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

