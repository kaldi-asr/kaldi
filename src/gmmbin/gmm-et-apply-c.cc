// gmmbin/gmm-et-apply-c.cc

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
        "Given the matrix Cpart which represents an MLLT/STC transform, "
        "let C = Cpart extended with zeros (and one as the new diagonal element); "
        "update the A and B matrices with A := C A C^{-1} and B := C B \n"
        "Usage:  gmm-et-apply-c [options] <et-object-in> <c-matrix-in> <et-object-out>\n"
        "e.g.:\n"
        " gmm-et-apply-c 1.et C.mat 2.et\n";

    bool binary = true;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string et_rxfilename = po.GetArg(1),
        cpart_rxfilename = po.GetArg(2),
        et_wxfilename = po.GetArg(3);

    ExponentialTransform et;
    {
      bool binary_in;
      Input ki(et_rxfilename, &binary_in);
      et.Read(ki.Stream(), binary_in);
    }
    Matrix<BaseFloat> Cpart;
    {
      bool binary_in;
      Input ki(cpart_rxfilename, &binary_in);
      Cpart.Read(ki.Stream(), binary_in);
    }
    et.ApplyC(Cpart);
    Output ko(et_wxfilename, binary);
    et.Write(ko.Stream(), binary);
    KALDI_LOG << "Applied C transform and wrote ET object.";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

