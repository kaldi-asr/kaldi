// latbin/lattice-add-penalty.cc

// Copyright 2013     Bagher BabaAli
//                    Johns Hopkins University (Author: Daniel Povey)

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

#include "lat/lattice-functions.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Add word insertion penalty to the lattice.\n"
        "Note: penalties are negative log-probs, base e, and are added to the\n"
        "'language model' part of the cost.\n"
        "\n"
        "Usage: lattice-add-penalty [options] <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-add-penalty --word-ins-penalty=1.0 ark:- ark:-\n";
      
    ParseOptions po(usage);
    
    BaseFloat word_ins_penalty = 0.0;

    po.Register("word-ins-penalty", &word_ins_penalty, "Word insertion penalty");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);
    
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    CompactLatticeWriter clat_writer(lats_wspecifier); // write as compact.

    int64 n_done = 0;

    for (; !clat_reader.Done(); clat_reader.Next()) {
      CompactLattice clat(clat_reader.Value());
      AddWordInsPenToCompactLattice(word_ins_penalty, &clat);
      clat_writer.Write(clat_reader.Key(), clat);
      n_done++;
    }
    KALDI_LOG << "Done adding word insertion penalty to " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
