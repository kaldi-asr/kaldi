// fstbin/fstpredeterminize.cc

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
#include "fst/fstlib.h"
#include "fstext/pre-determinize.h"
#include "fstext/fstext-utils.h"


/*
  example of test: (echo "0 0 0 0"; echo "0 0" ) | fstcompile | fstpredeterminize out.lst | fstprint

  Verifying correctness (of this, fstdeterminizestar and fstrmsymbols):

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

    const char *usage =
        "Predeterminizes input FST by adding input symbols as necessary for\n"
        "fstdeterminizestar to succeed.\n"
        "\n"
        "Usage:  fstpredeterminize disambig_out.list [in.fst [out.fst] ]\n";

    // no options.
    // bool binary = true;
    int32 first_disambig = 0;
    std::string prefix = "#";
    ParseOptions po(usage);
    po.Register("first-disambig", &first_disambig, "If nonzero, and on symbol-table present, ID of first disambiguation symbol");
    po.Register("prefix", &prefix, "Prefix of disambiguation symbols [if symbol-table provided.]");
    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string disambig_out_filename = po.GetArg(1),
        fst_in_filename = po.GetOptArg(2),
        fst_out_filename = po.GetOptArg(3);

    VectorFst<StdArc> *fst = ReadFstKaldi(fst_in_filename);

    std::vector<StdArc::Label> syms;

    int32 next_sym = 1 + HighestNumberedInputSymbol(*fst);
    if (first_disambig == 0)
      first_disambig = next_sym;
    else if (first_disambig < next_sym) {
      KALDI_WARN << "fstpredeterminize: warning: invalid first_disambig option given "
                 << first_disambig << " < " << next_sym << ", using " << next_sym;
      first_disambig = next_sym;
    }
    PreDeterminize(fst, static_cast<StdArc::Label>(first_disambig), &syms);

    // Output disambig symbols.
    if (!WriteIntegerVectorSimple(disambig_out_filename, syms))
      std::cerr << "fstpredeterminize: could not write disambig symbols to "<<
          (disambig_out_filename == "" ? "standard output" : disambig_out_filename.c_str())
                << '\n';

    WriteFstKaldi(*fst, fst_out_filename);
    delete fst;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

