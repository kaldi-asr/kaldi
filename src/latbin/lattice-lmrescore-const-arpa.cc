// latbin/lattice-lmrescore-const-arpa.cc

// Copyright 2014  Guoguo Chen

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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "lm/const-arpa-lm.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Rescores lattice with the ConstArpaLm format language model. The LM\n"
        "will be wrapped into the DeterministicOnDemandFst interface and the\n"
        "rescoring is done by composing with the wrapped LM using a special\n"
        "type of composition algorithm. Determinization will be applied on\n"
        "the composed lattice.\n"
        "\n"
        "Usage: lattice-lmrescore-const-arpa [options] lattice-rspecifier \\\n"
        "                                   const-arpa-in lattice-wspecifier\n"
        " e.g.: lattice-lmrescore-const-arpa --lm-scale=-1.0 ark:in.lats \\\n"
        "                                   const_arpa ark:out.lats\n";

    ParseOptions po(usage);
    BaseFloat lm_scale = 1.0;

    po.Register("lm-scale", &lm_scale, "Scaling factor for language model "
                "costs; frequently 1.0 or -1.0");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        lm_rxfilename = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);

    // Reads the language model in ConstArpaLm format.
    ConstArpaLm const_arpa;
    ReadKaldiObject(lm_rxfilename, &const_arpa);

    // Reads and writes as compact lattice.
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_done = 0, n_fail = 0;
    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      std::string key = compact_lattice_reader.Key();
      CompactLattice clat = compact_lattice_reader.Value();
      compact_lattice_reader.FreeCurrent();

      if (lm_scale != 0.0) {
        // Before composing with the LM FST, we scale the lattice weights
        // by the inverse of "lm_scale".  We'll later scale by "lm_scale".
        // We do it this way so we can determinize and it will give the
        // right effect (taking the "best path" through the LM) regardless
        // of the sign of lm_scale.
        fst::ScaleLattice(fst::GraphLatticeScale(1.0/lm_scale), &clat);
        ArcSort(&clat, fst::OLabelCompare<CompactLatticeArc>());

        // Wraps the ConstArpaLm format language model into FST. We re-create it
        // for each lattice to prevent memory usage increasing with time.
        ConstArpaLmDeterministicFst const_arpa_fst(const_arpa);

        // Composes lattice with language model.
        CompactLattice composed_clat;
        ComposeCompactLatticeDeterministic(clat,
                                           &const_arpa_fst, &composed_clat);

        // Determinizes the composed lattice.
        Lattice composed_lat;
        ConvertLattice(composed_clat, &composed_lat);
        Invert(&composed_lat);
        CompactLattice determinized_clat;
        DeterminizeLattice(composed_lat, &determinized_clat);
        fst::ScaleLattice(fst::GraphLatticeScale(lm_scale), &determinized_clat);
        if (determinized_clat.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty lattice for utterance " << key
              << " (incompatible LM?)";
          n_fail++;
        } else {
          compact_lattice_writer.Write(key, determinized_clat);
          n_done++;
        }
      } else {
        // Zero scale so nothing to do.
        n_done++;
        compact_lattice_writer.Write(key, clat);
      }
    }

    KALDI_LOG << "Done " << n_done << " lattices, failed for " << n_fail;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
