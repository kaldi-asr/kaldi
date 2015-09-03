// latbin/lattice-project.cc

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

    const char *usage =
        "Project lattices (in their transducer form); by default makes them\n"
        "word->word transducers (set --project-output=false for tid->tid).\n"
        "Usage: lattice-project [options] lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-project ark:1.lats ark:word2word.lats\n"
        "or: lattice-project --project-output=false ark:1.lats ark:tid2tid.lats";
        
    ParseOptions po(usage);
    bool project_output = true;
    
    po.Register("project-output", &project_output, "If true, project on output "
                "(words), else input (transition-ids)");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);

    LatticeWriter lattice_writer(lats_wspecifier); 
    int32 n_done = 0; // there is no failure mode, barring a crash.
      
    if (project_output) {
      SequentialCompactLatticeReader clat_reader(lats_rspecifier);
      for (; !clat_reader.Done(); clat_reader.Next()) {
        CompactLattice clat = clat_reader.Value();
        std::string key = clat_reader.Key();
        clat_reader.FreeCurrent();
        RemoveAlignmentsFromCompactLattice(&clat);
        Lattice lat;
        ConvertLattice(clat, &lat);
        fst::Project(&lat, fst::PROJECT_OUTPUT); // project on words.        
        lattice_writer.Write(key, lat);
        n_done++;
      }
    } else {
      // Read and write as regular lattice.
      SequentialLatticeReader lattice_reader(lats_rspecifier);
      for (; !lattice_reader.Done(); lattice_reader.Next()) {
        std::string key = lattice_reader.Key();
        Lattice lat = lattice_reader.Value();
        lattice_reader.FreeCurrent();
        fst::Project(&lat, fst::PROJECT_INPUT);
        lattice_writer.Write(key, lat);
        n_done++;
      }
    }
    KALDI_LOG << "Done projecting " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
