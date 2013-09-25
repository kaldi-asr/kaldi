// gmmbin/gmm-et-get-a.cc

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
        "Write out the A matrix from the exponential transform\n"
        " to a single file\n"
        "Usage:  gmm-et-get-a [options] <et-object-in> <matrix-out>\n"
        "e.g.: \n"
        " gmm-et-get-a --binary=false 12.et A.mat\n";

    bool binary = true;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string et_rxfilename = po.GetArg(1),
        a_wxfilename = po.GetArg(2);
    
    ExponentialTransform et;
    ReadKaldiObject(et_rxfilename, &et);
    Matrix<BaseFloat> A;
    et.GetDefaultTransform(&A);
    Output ko(a_wxfilename, binary);
    A.Write(ko.Stream(), binary);
    KALDI_LOG << "Wrote A  to " << a_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

