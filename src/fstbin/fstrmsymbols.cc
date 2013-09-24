// fstbin/fstrmsymbols.cc

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
#include "fst/fstlib.h"
#include "fstext/determinize-star.h"
#include "fstext/fstext-utils.h"


/* some test examples:
 ( echo 3; echo  4) > /tmp/in.list
 ( echo "0 0 1 1"; echo " 0 0 3 2"; echo "0 0"; ) | fstcompile | fstrmsymbols /tmp/in.list | fstprint

  cd ~/tmpdir
  while true; do
    fstrand > 1.fst
    fstpredeterminize out.lst 1.fst | fstdeterminizestar | fstrmsymbols out.lst > 2.fst
    fstequivalent --random=true 1.fst 2.fst || echo "Test failed"
    echo -n "."
  done

*/

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    bool remove_from_output = false;
    
    const char *usage =
        "Replaces a subset of symbols with epsilon, wherever they appear on the input side\n"
        "of an FST (or the output side, with --remove-from-output=true)\n"
        "\n"
        "Usage:  fstrmsymbols in-disambig-list  [in.fst [out.fst] ]\n"
        "E.g:  fstrmsymbols in.list  < in.fst > out.fst\n";

    ParseOptions po(usage);
    po.Register("remove-from-output", &remove_from_output, "If true, remove these symbols from "
                "the output, not the input, side.");
    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string disambig_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetOptArg(2),
        fst_wxfilename = po.GetOptArg(3);

    VectorFst<StdArc> *fst = ReadFstKaldi(fst_rxfilename);
    
    std::vector<int32> disambig_in;
    if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_in))
      KALDI_ERR << "fstrmsymbols: Could not read disambiguation symbols from "
                << (disambig_rxfilename == "" ? "standard input" : disambig_rxfilename);

    if (remove_from_output) Invert(fst);
    RemoveSomeInputSymbols(disambig_in, fst);
    if (remove_from_output) Invert(fst);
    
    WriteFstKaldi(*fst, fst_wxfilename);

    delete fst;
    return 0;    
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

