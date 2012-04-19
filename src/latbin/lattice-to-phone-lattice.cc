// latbin/lattice-to-phone-lattice.cc

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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Convert the words (or whatever other output labels are on the lattice)\n"
        "into phones, worked out from the alignment (needs the transition model\n"
        "to work out the phone sequence from the transition-ids)\n"
        "Usage: lattice-to-phone-lattice [options] model lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-to-phone-lattice 1.mdl ark:1.lats ark:phones.lats\n";
      
    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        model_rxfilename = po.GetArg(1),
        lats_rspecifier = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);
    
    int32 n_done = 0;

    TransitionModel trans_model;
    
    ReadKaldiObject(model_rxfilename, &trans_model);
    
    SequentialLatticeReader lattice_reader(lats_rspecifier); // read as
    // regular lattice.
    CompactLatticeWriter clat_writer(lats_wspecifier); // write as compact.
    for (; !lattice_reader.Done(); lattice_reader.Next(), n_done++) {
      Lattice lat(lattice_reader.Value());
      ConvertLatticeToPhones(trans_model, &lat);
      CompactLattice clat;
      ConvertLattice(lat, &clat);
      clat_writer.Write(lattice_reader.Key(), clat);
      n_done++;
    }
    KALDI_LOG << "Done copying " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
