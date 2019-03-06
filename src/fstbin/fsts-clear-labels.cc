// fstbin/fsts-clear-symbols.cc

// Copyright 2019  Johns Hopkins University (Authors: Daniel Povey)
//           2019  Yiming Wang

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


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    typedef kaldi::int32 int32;
    typedef kaldi::uint64 uint64;

    const char *usage =
        "Reads kaldi archive of FSTs; for each element, sets all the symbols on\n"
        "the input and/or output side of the FST to zero, as specified.\n"
        "It does not alter the symbol tables.\n"
        "\n"
        "Usage: fsts-clear-labels [options] <fsts-rspecifier> <fsts-wspecifier>\n"
        " e.g.: fsts-clear-label ark:train.fsts ark,t:train.fsts\n";

    ParseOptions po(usage);

    bool clear_input = true, clear_output = true;

    po.Register("clear-input", &clear_input, "If true, clear input");
    po.Register("clear-output", &clear_output, "If true, clear output");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fsts_rspecifier = po.GetArg(1),
        fsts_wspecifier = po.GetArg(2);


    SequentialTableReader<VectorFstHolder> fst_reader(fsts_rspecifier);
    TableWriter<VectorFstHolder> fst_writer(fsts_wspecifier);

    int32 n_done = 0;
    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string key = fst_reader.Key();
      VectorFst<StdArc> fst(fst_reader.Value());

      ClearSymbols(clear_input, clear_output, &fst);

      fst_writer.Write(key, fst);
      n_done++;
    }

    KALDI_LOG << "Cleared " << n_done << " FSTs";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
