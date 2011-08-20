// gmmbin/gmm-et-get-b.cc

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
        "Write out the B matrix from the exponential transform (excluding last row), to a single file\n"
        "This can be treated as the \"default\" value of the exponential transform.\n"
        "Optionally, writes out the ET object with the B transform set to unity\n"
        " Usage:  gmm-et-get-b [options] <et-object-in> <matrix-out> [<et-object-out>]\n"
        "e.g.: \n"
        " gmm-et-get-b 12.et B.mat final.et\n";

    bool binary = true;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string et_rxfilename = po.GetArg(1),
        b_wxfilename = po.GetArg(2),
        et_wxfilename = po.GetOptArg(3);
    
    ExponentialTransform et;
    {
      bool binary_in;
      Input ki(et_rxfilename, &binary_in);
      et.Read(ki.Stream(), binary_in);
    }
    Matrix<BaseFloat> B;
    et.GetDefaultTransform(&B);
    Output ko(b_wxfilename, binary);
    B.Write(ko.Stream(), binary);
    KALDI_LOG << "Wrote B  to " << b_wxfilename;
    if (et_wxfilename != "") {
      Output ko(et_wxfilename, binary);
      et.MakeBUnit();
      et.Write(ko.Stream(), binary);
      KALDI_LOG << "Wrote ET to " << et_wxfilename;
    }
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

