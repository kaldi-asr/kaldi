// fstbin/fstrhocompose.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

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
#include "fstext/table-matcher.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"


/*
The following commands represent a basic test of this program.

cat <<EOF | fstcompile > a.fst
0  1  10 10
0  1  11 11
1
EOF

cat <<EOF | fstcompile > g.fst
0  0  11 11  0.0
0  1  100 100  1.0
1  0  11  11  1.0
1
EOF
fstcompose a.fst g.fst | fstprint
# gives, as expected, the empty FST.

fstrhocompose 100 a.fst g.fst | fstprint
# gives, again correctly:
#0    1    10    10    1
#1
 

*/

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;
    /*
      fstrhocompose does composition, but treats the second FST
      specially; the symbol "rho" is taken whenever the
      composition algorithm fails to find a match for a label
      in the second FST.  We rewrite "rho" with whatever label on the
      first FST we matched.   This is done on both sides of the output
      if the rho FST was an acceptor, and just on the input side otherwise
      (I think)... but typically the rho FST will be an acceptor.
      We can add options for this later if needed.
    */
      
    const char *usage =
        "Composition, where the right FST has \"rest\" (rho) transition\n"
        "that are only taken where there was no match of a \"real\" label\n"
        "You supply the label corresponding to rho.\n"
        "\n"
        "Usage:  fstrhocompose rho-label (fst1-rxfilename|fst1-rspecifier) "
        "(fst2-rxfilename|fst2-rspecifier) [(out-rxfilename|out-rspecifier)]\n"
        "E.g.: fstrhocompose 54 a.fst b.fst c.fst\n"
        "or: fstrhocompose 11 ark:a.fsts G.fst ark:b.fsts\n";
        
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        rho_str = po.GetArg(1),
        fst1_in_str = po.GetArg(2),
        fst2_in_str = po.GetArg(3),
        fst_out_str = po.GetOptArg(4);
    
    bool is_table_1 =
        (ClassifyRspecifier(fst1_in_str, NULL, NULL) != kNoRspecifier),
        is_table_2 =
        (ClassifyRspecifier(fst2_in_str, NULL, NULL) != kNoRspecifier),
        is_table_out =
        (ClassifyWspecifier(fst_out_str, NULL, NULL, NULL) != kNoWspecifier);

    int32 rho_label;
    if (!ConvertStringToInteger(rho_str, &rho_label)
        || rho_label <= 0)
      KALDI_ERR << "Invalid first argument (rho label), expect positive integer.";
    
    if (is_table_out != (is_table_1 || is_table_2))
      KALDI_ERR << "Incompatible combination of archives and files";
    
    if (!is_table_1 && !is_table_2) { // Only dealing with files...
      VectorFst<StdArc> *fst1 = ReadFstKaldi(fst1_in_str);
      
      VectorFst<StdArc> *fst2 = ReadFstKaldi(fst2_in_str);

      VectorFst<StdArc> composed_fst;

      RhoCompose(*fst1, *fst2, rho_label, &composed_fst);
      
      delete fst1;
      delete fst2;

      WriteFstKaldi(composed_fst, fst_out_str);
      return 0;
    } else if (is_table_1 && !is_table_2) {

      VectorFst<StdArc> *fst2 = ReadFstKaldi(fst2_in_str);

      SequentialTableReader<VectorFstHolder> fst1_reader(fst1_in_str);
      TableWriter<VectorFstHolder> fst_writer(fst_out_str);
      int32 n_done = 0;
      for (; !fst1_reader.Done(); fst1_reader.Next(), n_done++) {
        VectorFst<StdArc> fst1(fst1_reader.Value());
        VectorFst<StdArc> fst_out;
        RhoCompose(fst1, *fst2, rho_label, &fst_out);
        fst_writer.Write(fst1_reader.Key(), fst_out);
      }
      KALDI_LOG << "Composed " << n_done << " FSTs.";
      return (n_done != 0 ? 0 : 1);
    } else {
      KALDI_ERR << "The combination of tables/non-tables that you "
                << "supplied is not currently supported.  Either implement this, "
                << "ask the maintainers to implement it, or call this program "
                << "differently.";
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
