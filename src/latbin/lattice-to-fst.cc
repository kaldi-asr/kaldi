// latbin/lattice-to-fst.cc

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
    using std::vector;
    BaseFloat acoustic_scale = 0.0;
    BaseFloat lm_scale = 0.0;
    bool rm_eps = true;
    
    const char *usage =
        "Turn lattices into normal FSTs, retaining only the word labels\n"
        "By default, removes all weights and also epsilons (configure with\n"
        "with --acoustic-scale, --lm-scale and --rm-eps)\n"
        "Usage: lattice-to-fst [options] lattice-rspecifier fsts-wspecifier\n"
        " e.g.: lattice-to-fst  ark:1.lats ark:1.fsts\n";
      
    ParseOptions po(usage);
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale, "Scaling factor for graph/lm costs");
    po.Register("rm-eps", &rm_eps, "Remove epsilons in resulting FSTs (in lazy way; may not remove all)");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    vector<vector<double> > scale = fst::LatticeScale(lm_scale, acoustic_scale);
    
    std::string lats_rspecifier = po.GetArg(1),
        fsts_wspecifier = po.GetArg(2);
    
    SequentialCompactLatticeReader lattice_reader(lats_rspecifier);
    TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);
    
    int32 n_done = 0; // there is no failure mode, barring a crash.
    
    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      CompactLattice clat = lattice_reader.Value();
      lattice_reader.FreeCurrent();
      ScaleLattice(scale, &clat); // typically scales to zero.
      RemoveAlignmentsFromCompactLattice(&clat); // remove the alignments...
      fst::VectorFst<StdArc> fst;
      {
        Lattice lat;
        ConvertLattice(clat, &lat); // convert to non-compact form.. won't introduce
        // extra states because already removed alignments.
        ConvertLattice(lat, &fst); // this adds up the (lm,acoustic) costs to get
        // the normal (tropical) costs.
        Project(&fst, fst::PROJECT_OUTPUT); // Because in the standard Lattice format,
        // the words are on the output, and we want the word labels.
      }
      if (rm_eps) RemoveEpsLocal(&fst);
      
      fst_writer.Write(key, fst);
      n_done++;
    }
    KALDI_LOG << "Done converting " << n_done << " lattices to word-level FSTs";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
