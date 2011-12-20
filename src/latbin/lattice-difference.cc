// latbin/lattice-rmnum.cc

// Copyright 2009-2011 Chao Weng 

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
        "remove transcription word sequence from denominator lattice\n"
        "Mainly for the denominator lattice for MCE.\n"    
        "Usage: lattice-difference [option] transcriptions-rspecifier"
        " lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-rmnum ark:train.tra ark:den.lats ark:den_mce.lats\n"; 

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
  
    std::string transcript_rspecifier = po.GetArg(1);
    std::string lats_rspecifier = po.GetArg(2);
    std::string lats_wspecifier = po.GetArg(3);

    SequentialInt32VectorReader transcript_reader(transcript_rspecifier);
    RandomAccessCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_removed = 0, n_no_lat = 0, n_only_transcription = 0;

    for (; !transcript_reader.Done(); transcript_reader.Next()) {
      std::string key = transcript_reader.Key();
      const std::vector<int32> &transcript = transcript_reader.Value();
      CompactLattice transcript_fst;
      MakeLinearAcceptor(transcript, &transcript_fst);
      if (compact_lattice_reader.HasKey(key)) {
        const CompactLattice &clat = compact_lattice_reader.Value(key);
        CompactLattice clat_out;
        Difference(clat, transcript_fst, &clat_out);
        if (clat_out.Start() == 0) {
          compact_lattice_writer.Write(key, clat_out);
          n_removed++; 
        } else {
          //fall in the case the lattice only contains transcription 
          n_only_transcription++;
        }
      } else {
        KALDI_WARN << "No lattice found for utterance " << key << " in "
                   << lats_rspecifier;
        n_no_lat++;
      }
    }
    
    KALDI_LOG << "Total " << n_removed << "lattices written."
              << n_only_transcription
              << " lattices contain only transcription Missing lattices in "
              << n_no_lat << " cases.";
    return (n_removed != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
