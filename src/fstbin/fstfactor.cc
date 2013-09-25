// fstbin/fstfactor.cc

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
#include "util/kaldi-io.h"
#include "util/parse-options.h"
#include "util/text-utils.h"
#include "fst/fstlib.h"
#include "fstext/table-matcher.h"
#include "fstext/fstext-utils.h"


/*
  cd ~/tmpdir
  while true; do
    fstrand  > 1.fst
    cat 1.fst |  fstfactor - 2.fst - > 3.fst # just checking pipes work.
    fstarcsort --sort_type=olabel 2.fst | fsttablecompose - 3.fst  > 1b.fst
    fstequivalent --random=true 1.fst 1b.fst || echo "Test failed"
    echo -n "."
  done

*/

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;
    /*
      fstfactor in.fst out1.fst out2.fst
      produces two fsts such that the composition of out1.fst and out2.fst
      is equivalent to in.fst.  It does this by representing linear chains of
      input symbols in in.fst, as special symbols that will be on the output
      of out1.fst (and the input of out2.fst).

      out1.fst has a simple structure with a loop-state that's initial and final,
      and output symbols leading to linear chains of input symbols that come back
      to the loop state.
    */

    const char *usage =
        "Factor fst into two parts (by removing linear chains)\n"
        "\n"
        "Usage:  fstfactor in.fst out1.fst out2.fst\n";

    ParseOptions po(usage);

    bool push = false;

    po.Register("push", &push,
                "Push output symbols to initial state before factoring");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }


    std::string fst_in_filename = po.GetArg(1),
        fst1_out_filename = po.GetArg(2),
        fst2_out_filename = po.GetArg(3);

    VectorFst<StdArc> *fst = ReadFstKaldi(fst_in_filename);

    if (push) {
      KALDI_VLOG(1) <<  "Pushing symbols\n";
      VectorFst<StdArc> fst_pushed;
      Push<StdArc, REWEIGHT_TO_INITIAL>(*fst, &fst_pushed, kPushLabels, kDelta);
      *fst = fst_pushed;
      KALDI_VLOG(1) <<  "Factoring\n";
    }

    VectorFst<StdArc> fst1, fst2;
    Factor(*fst, &fst1, &fst2);  // int32 is enough for forseeable uses..

    delete fst;

    WriteFstKaldi(fst1, fst1_out_filename);
    WriteFstKaldi(fst2, fst2_out_filename);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

