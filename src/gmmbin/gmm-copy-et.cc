// gmmbin/gmm-copy-et.cc

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
        "Copy exponential transform object (possibly changing normalization type)\n"
        "Usage:  gmm-copy-et [options] <et-object-in> <et-object-out>\n"
        "e.g.: \n"
        " gmm-copy-et --normalize-type=mean-and-var  1.et 2.et\n";

    bool binary = true;
    std::string normalize_type = "";

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("normalize-type", &normalize_type, "Change normalization type: \"\"|\"mean\"|\"mean-and-var\"|\"none\"");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string et_rxfilename = po.GetArg(1),
        et_wxfilename = po.GetArg(2);

    ExponentialTransform et;
    {
      bool binary_in;
      Input ki(et_rxfilename, &binary_in);
      et.Read(ki.Stream(), binary_in);
    }

    if (normalize_type != "") {
      EtNormalizeType nt;
      if (normalize_type == "offset") nt = kEtNormalizeOffset;
      else if (normalize_type == "diag") nt = kEtNormalizeDiag;
      else if (normalize_type == "none") nt = kEtNormalizeNone;
      // "none" unlikely, since pointless: only allowed if already == none.
      else KALDI_ERR << "Invalid normalize-type option: " << normalize_type;
      // The next statement may fail if you tried to reduce
      // the amount of normalization.
      et.SetNormalizeType(nt);
    }

    Output ko(et_wxfilename, binary);
    et.Write(ko.Stream(), binary);
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

