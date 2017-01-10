// fstbin/fsts-difference.cc

// Copyright 2016  Johns Hopkins University (Authors: Jan "Yenda" Trmal)

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
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"


int main(int argc, char **argv) {
  try {
    using namespace kaldi;
    using namespace fst;
    typedef kaldi::int32 int32;
    typedef kaldi::uint64 uint64;

    const char *usage =
        "Reads a table of FSTs; for each element, performs the fst subtract\n"
        "operation. This operation computes the difference between two FSAs.\n"
        "Only strings that are in the first automaton but not in second are\n"
        "retained in the result.\n"
        "\n"
        "Usage: fsts-subtract [options] <fsts-rspecifier> "
                                        "<fsts-rspecifier> "
                                        "<fsts-wspecifier>\n"
        " e.g.: fsts-subtract ark:A.fsts ark:B.fsts ark,t:C.fsts\n";

    ParseOptions po(usage);

    bool project_output = false;

    po.Register("project-output", &project_output,
                "If true, project output vs input");

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string fsts_rspecifier1 = po.GetArg(1),
                fsts_rspecifier2 = po.GetArg(2),
                fsts_wspecifier = po.GetArg(3);


    SequentialTableReader<VectorFstHolder> fst_reader1(fsts_rspecifier1);
    RandomAccessTableReader<VectorFstHolder> fst_reader2(fsts_rspecifier2);
    TableWriter<VectorFstHolder> fst_writer(fsts_wspecifier);

    int32 n_done = 0, n_skipped = 0;
    for (; !fst_reader1.Done(); fst_reader1.Next()) {
      std::string key = fst_reader1.Key();
      const VectorFst<StdArc> &A(fst_reader1.Value());

      if (fst_reader2.HasKey(key)) {
        const VectorFst<StdArc> &B(fst_reader2.Value(key));
        VectorFst<StdArc> C;

        Difference(A, B, &C);
        fst_writer.Write(key, C);
      } else {
        fst_writer.Write(key, A);
        n_skipped++;
      }

      n_done++;
    }

    KALDI_LOG << "Processed " << n_done
              << " FSTs, skipped " << n_skipped << "FSTs";

    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
