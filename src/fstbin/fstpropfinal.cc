// fstbin/fstpropfinal.cc

// Copyright 2009-2011  Microsoft Corporation
//                2016  Johns Hopkins University (author: Daniel Povey)

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
#include "fst/fstlib.h"
#include "fstext/determinize-star.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"

/* A test  example.
   You have to have the right things on your PATH for this to work.

cat <<EOF | fstcompile | fstpropfinal 10 | fstprint
0 1 5 5 0.0
0 1 10 10 5.0
1 2 10 10 10.0
2
EOF

*/


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Propagates final-states through phi transitions\n"
        "\n"
        "Usage:  fstpropfinal phi-label [in.fst [out.fst] ]\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() > 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string phi_str = po.GetOptArg(1),
        fst_in_str = po.GetOptArg(2),
        fst_out_str = po.GetOptArg(3);


    int32 phi_label;
    if (!ConvertStringToInteger(phi_str, &phi_label)
        || phi_label < 0)
      KALDI_ERR << "Bad phi label " << phi_label;
    if (phi_label == 0)
      KALDI_WARN  << "Phi_label == 0, may not be a good idea.";


    VectorFst<StdArc> *fst = ReadFstKaldi(fst_in_str);

    PropagateFinal(phi_label, fst);

    WriteFstKaldi(*fst, fst_out_str);
    delete fst;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
