// fstbin/fstdeterminizestar.cc

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
#ifndef _MSC_VER
#include <signal.h> // Comment this line and the call to signal below if
// it causes compilation problems.  It is only to enable a debugging procedure
// when determinization does not terminate.  
#endif

/* some test  examples:
 ( echo "0 0 0 0"; echo "0 0" ) | fstcompile | fstdeterminizestar | fstprint
 ( echo "0 0 1 0"; echo "0 0" ) | fstcompile | fstdeterminizestar | fstprint
 ( echo "0 0 1 0"; echo "0 1 1 0"; echo "0 0" ) | fstcompile | fstdeterminizestar | fstprint
 # this last one fails [correctly]:
 ( echo "0 0 0 1"; echo "0 0" ) | fstcompile | fstdeterminizestar | fstprint

  cd ~/tmpdir
  while true; do
    fstrand > 1.fst
    fstpredeterminize out.lst 1.fst | fstdeterminizestar | fstrmsymbols out.lst > 2.fst
    fstequivalent --random=true 1.fst 2.fst || echo "Test failed"
    echo -n "."
  done

 Test of debugging [with non-determinizable input]:
 ( echo " 0 0 1 0 1.0"; echo "0 1 1 0"; echo "1 1 1 0 0"; echo "0 2 2 0"; echo "2"; echo "1" ) | fstcompile | fstdeterminizestar
  kill -SIGUSR1 [the process-id of fstdeterminizestar]
  # prints out a bunch of debugging output showing the mess it got itself into.
*/


bool debug_location = false;
void signal_handler(int) {
  debug_location = true;
}



int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Removes epsilons and determinizes in one step\n"
        "\n"
        "Usage:  fstdeterminizestar [in.fst [out.fst] ]\n";

    float delta = kDelta;
    int max_states = -1;
    bool use_log = false;
    ParseOptions po(usage);
    po.Register("use-log", &use_log, "Determinize in log semiring.");
    po.Register("delta", &delta, "Delta value used to determine equivalence of weights.");
    po.Register("max-states", &max_states, "Maximum number of states in determinized FST before it will abort.");
    po.Read(argc, argv);

    if (po.NumArgs() > 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fst_in_filename;
    fst_in_filename = po.GetOptArg(1);
    if (fst_in_filename == "-") fst_in_filename = "";

    std::string fst_out_filename;
    fst_out_filename = po.GetOptArg(2);
    if (fst_out_filename == "-") fst_out_filename = "";

    // This enables us to get traceback info from determinization that is
    // not seeming to terminate.
#ifndef _MSC_VER
    signal(SIGUSR1, signal_handler);
#endif

    VectorFst<StdArc> *fst = VectorFst<StdArc>::Read(fst_in_filename);
    if (!fst) {
      std::cerr << "fstdeterminizestar: could not read input fst from "
                << fst_in_filename << '\n';
      return 1;
    }

    ArcSort(fst, ILabelCompare<StdArc>());  // helps DeterminizeStar to be faster.      
    if (use_log) {
      DeterminizeStarInLog(fst, delta, &debug_location, max_states);
    } else {
      VectorFst<StdArc> det_fst;
      DeterminizeStar(*fst, &det_fst, delta, &debug_location, max_states);
      *fst = det_fst;  // will do shallow copy and then det_fst goes out of scope anyway.
    }
    if (! fst->Write(fst_out_filename) ) {
      std::cerr << "fstdeterminizestar: error writing the output to "<<fst_out_filename << '\n';
      return 1;
    }
    delete fst;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}

