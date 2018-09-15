// latbin/lattice-reverse.cc

// Copyright        2018 Hainan Xu

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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Reverse a lattice in order to rescore the lattice with a RNNLM \n"
        "trained reversed text. An example for its application is at \n"
        "swbd/local/rnnlm/run_lstm_tdnn_back.sh\n"
        "Usage: lattice-reverse lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-reverse ark:forward.lats ark:backward.lats\n";
    
    ParseOptions po(usage);
    std::string include_rxfilename;
    std::string exclude_rxfilename;

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
                lats_wspecifier = po.GetArg(2);

    int32 n_done = 0;
    
    SequentialLatticeReader lattice_reader(lats_rspecifier);
    LatticeWriter lattice_writer(lats_wspecifier);
    
    for (; !lattice_reader.Done(); lattice_reader.Next(), n_done++) {
      string key = lattice_reader.Key();
      Lattice &lat = lattice_reader.Value();
      Lattice olat;
      fst::Reverse(lat, &olat);
      lattice_writer.Write(lattice_reader.Key(), olat);
    }

    KALDI_LOG << "Done reversing " << n_done << " lattices.";
    
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
