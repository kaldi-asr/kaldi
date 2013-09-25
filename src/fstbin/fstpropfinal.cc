// fstbin/fstpropfinal.cc

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
#ifndef _MSC_VER
#include <signal.h> // Comment this line and the call to signal below if
// it causes compilation problems.  It is only to enable a debugging procedure
// when determinization does not terminate.  
#endif

/* some test  examples.
   You have to have the right things on your PATH for this to work.

cat <<EOF | fstcompile | fstpropfinal 10 | fstprint
0 1 5 5 0.0
0 1 10 10 5.0
1 2 10 10 10.0
2
EOF


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

