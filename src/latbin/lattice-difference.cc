// latbin/lattice-difference.cc

// Copyright 2009-2011 Chao Weng 

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
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Compute FST difference on lattices (remove sequences in first lattice\n"
        " that appear in second lattice)\n"
        "Useful for the denominator lattice for MCE.\n"    
        "Usage: lattice-difference [options] "
        "lattice1-rspecifier lattice2-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-difference ark:den.lats ark:num.lats ark:den_mce.lats\n"; 

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
  
    std::string lats1_rspecifier = po.GetArg(1);
    std::string lats2_rspecifier = po.GetArg(2);
    std::string lats_wspecifier = po.GetArg(3);

    SequentialCompactLatticeReader compact_lattice_reader1(lats1_rspecifier);
    RandomAccessCompactLatticeReader compact_lattice_reader2(lats2_rspecifier);
    
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_done = 0, n_no_lat = 0, n_only_transcription = 0;

    for (; !compact_lattice_reader1.Done(); compact_lattice_reader1.Next()) {
      std::string key = compact_lattice_reader1.Key();
      const CompactLattice &clat1 =  compact_lattice_reader1.Value();
      if (compact_lattice_reader2.HasKey(key)) {
        CompactLattice clat2 (compact_lattice_reader2.Value(key));
        // "Difference" requires clat2 to be unweighted, deterministic and epsilon-free.
        // So we remove the weights, remove epsilons and determinize.
        RemoveWeights(&clat2);
        RmEpsilon(&clat2);
        { CompactLattice clat_tmp(clat2); Determinize(clat_tmp, &clat2); }
        
        CompactLattice clat_out;
        Difference(clat1, clat2, &clat_out);
        if (clat_out.Start() == 0) {
          compact_lattice_writer.Write(key, clat_out);
          n_done++; 
        } else {
          // In this case, the lattice only contains the transcription
          KALDI_WARN << "Skipping utterance " << key
                     << " because difference is empty.";
          n_only_transcription++;
        }
      } else {
        KALDI_WARN << "No lattice found for utterance " << key << " in "
                   << lats2_rspecifier;
        n_no_lat++;
      }
    }
    
    KALDI_LOG << "Total " << n_done << " lattices written; "
              << n_only_transcription
              << " lattices had empty difference; "
              << n_no_lat << " missing lattices in second archive ";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
