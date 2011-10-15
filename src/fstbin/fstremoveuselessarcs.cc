// fstbin/fstremoveuselessarcs.cc

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
#include "util/kaldi-io.h"
#include "util/parse-options.h"
#include "util/text-utils.h"
#include "fst/fstlib.h"
#include "fstext/determinize-star.h"
#include "fstext/fstext-utils.h"


/* some test  examples:
 ( echo "0 0 0 0"; echo "0" ) | fstcompile | fstremoveuselessarcs | fstprint
 # just prints "0" (removes the self-loop)
 ( echo "0 1 1 2"; echo " 0 1 1 3"; echo "1 0";  ) | fstcompile | fstremoveuselessarcs | fstprint
  # just prints 0 1 1 2 \n 1 [it removes an arbitrarily chosen arc]
  ( echo "0 1 1 2 1.0"; echo " 0 1 1 3"; echo "1 0";  ) | fstcompile | fstremoveuselessarcs | fstprint
  # prints  0	1	1	3 \n  1
  ( echo "0 1 1 2 1.0"; echo " 0 1 1 3 2.0"; echo "1 0";  ) | fstcompile | fstremoveuselessarcs | fstprint
  # prints 0	1	1	2	1 \n 1

*/

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Remove useless arcs from an FST (those which will never be on a best "
        "path for any input sequence; also removes ties)."
        "\n"
        "Usage:  fstremoveuselessarcs [in.fst [out.fst] ]\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() > 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fst_in_filename = po.GetOptArg(1),
        fst_out_filename = po.GetOptArg(2);

    VectorFst<StdArc> *fst = ReadFstKaldi(fst_in_filename);

    RemoveUselessArcs(fst);

    WriteFstKaldi(*fst, fst_out_filename);
    delete fst;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

