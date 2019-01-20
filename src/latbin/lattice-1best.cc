// latbin/lattice-1best.cc

// Copyright 2009-2012  Stefan Kombrink  Johns Hopkins University (Author: Daniel Povey)
//           2018       Music Technology Group, Universitat Pompeu Fabra (Rong Gong)


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
#include "lat/lattice-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Compute best path through lattices and write out as FSTs\n"
        "Note: differs from lattice-nbest with --n=1 because we won't\n"
        "append -1 to the utterance-ids.  Differs from lattice-best-path\n"
        "because output is FST.\n"
        "\n"
        "Usage: lattice-1best [options] <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-1best --acoustic-scale=0.1 ark:1.lats ark:1best.lats\n";
      
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0;
    BaseFloat lm_scale = 1.0;
    BaseFloat word_ins_penalty = 0.0;
    
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale,
                "Scaling factor for language model scores.");
    po.Register("word-ins-penalty", &word_ins_penalty,
                "Word insertion penality.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);

    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    
    // Write as compact lattice.
    CompactLatticeWriter compact_1best_writer(lats_wspecifier); 

    int32 n_done = 0, n_err = 0;

    if (acoustic_scale == 0.0 || lm_scale == 0.0)
      KALDI_ERR << "Do not use exactly zero acoustic or LM scale (cannot be inverted)";
    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();
      clat_reader.FreeCurrent();
      fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);
      if (word_ins_penalty > 0.0) {
        AddWordInsPenToCompactLattice(word_ins_penalty, &clat);
      }

      CompactLattice best_path;
      CompactLatticeShortestPath(clat, &best_path);
      
      if (best_path.Start() == fst::kNoStateId) {
        KALDI_WARN << "Possibly empty lattice for utterance-id " << key
                   << "(no output)";
        n_err++;
      } else {
        if (word_ins_penalty > 0.0) {
          AddWordInsPenToCompactLattice(-word_ins_penalty, &best_path);
        }
        fst::ScaleLattice(fst::LatticeScale(1.0 / lm_scale, 1.0/acoustic_scale),
                          &best_path);
        compact_1best_writer.Write(key, best_path);
        n_done++;
      }
    }
    KALDI_LOG << "Done converting " << n_done << " to best path, "
              << n_err << " had errors.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
