// latbin/lattice-reverse.cc

// Copyright 2012 BUT Mirko Hannemann

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
//#include "lat/lattice-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Time reversal of compact lattice and write out as lattice\n"
        "Usage: lattice-reverse [options] lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-reverse ark:1.lats ark:1.reverse.lats\n";
      
    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);

    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    
    // Write as compact lattice.
    CompactLatticeWriter compact_lat_writer(lats_wspecifier); 

    int32 n_done = 0;

    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();
      clat_reader.FreeCurrent();
      
      Lattice lat;
      ConvertLattice(clat, &lat);
      Lattice reverse_lat;
      fst::Reverse(lat, &reverse_lat);
      RemoveEpsLocal(&reverse_lat);
      CompactLattice reverse_clat;
      ConvertLattice(reverse_lat, &reverse_clat);
      RemoveEpsLocal(&reverse_clat);
    
      compact_lat_writer.Write(key, reverse_clat);
      n_done++;
    }
    KALDI_LOG << "Done converting " << n_done << " to best path";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
