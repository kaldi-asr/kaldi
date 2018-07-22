// latbin/lattice-lmrescore-rnnlm.cc

// Copyright 2015  Guoguo Chen

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
#include "lm/kaldi-rnnlm.h"
#include "lm/mikolov-rnnlm-lib.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Rescores lattice with rnnlm. The LM will be wrapped into the\n"
        "DeterministicOnDemandFst interface and the rescoring is done by\n"
        "composing with the wrapped LM using a special type of composition\n"
        "algorithm. Determinization will be applied on the composed lattice.\n"
        "\n"
        "Usage: lattice-lmrescore-rnnlm [options] [unk_prob_rspecifier] \\\n"
        "             <word-symbol-table-rxfilename> <lattice-rspecifier> \\\n"
        "             <rnnlm-rxfilename> <lattice-wspecifier>\n"
        " e.g.: lattice-lmrescore-rnnlm --lm-scale=-1.0 words.txt \\\n"
        "                     ark:in.lats rnnlm ark:out.lats\n";

    ParseOptions po(usage);
    int32 max_ngram_order = 3;
    BaseFloat lm_scale = 1.0;

    po.Register("lm-scale", &lm_scale, "Scaling factor for language model "
                "costs; frequently 1.0 or -1.0");
    po.Register("max-ngram-order", &max_ngram_order, "If positive, limit the "
                "rnnlm context to the given number, -1 means we are not going "
                "to limit it.");

    KaldiRnnlmWrapperOpts opts;
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4 && po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier, unk_prob_rspecifier,
        word_symbols_rxfilename, rnnlm_rxfilename, lats_wspecifier;
    if (po.NumArgs() == 4) {
      unk_prob_rspecifier = "";
      word_symbols_rxfilename = po.GetArg(1);
      lats_rspecifier = po.GetArg(2);
      rnnlm_rxfilename = po.GetArg(3);
      lats_wspecifier = po.GetArg(4);
    } else if (po.NumArgs() == 5) {
      unk_prob_rspecifier = po.GetArg(1);
      word_symbols_rxfilename = po.GetArg(2);
      lats_rspecifier = po.GetArg(3);
      rnnlm_rxfilename = po.GetArg(4);
      lats_wspecifier = po.GetArg(5);
    }

    // Reads the language model.
    KaldiRnnlmWrapper rnnlm(opts, unk_prob_rspecifier,
                            word_symbols_rxfilename, rnnlm_rxfilename);

    // Reads and writes as compact lattice.
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_done = 0, n_fail = 0;
    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      std::string key = compact_lattice_reader.Key();
      CompactLattice &clat = compact_lattice_reader.Value();

      if (lm_scale != 0.0) {
        // Before composing with the LM FST, we scale the lattice weights
        // by the inverse of "lm_scale".  We'll later scale by "lm_scale".
        // We do it this way so we can determinize and it will give the
        // right effect (taking the "best path" through the LM) regardless
        // of the sign of lm_scale.
        fst::ScaleLattice(fst::GraphLatticeScale(1.0 / lm_scale), &clat);
        ArcSort(&clat, fst::OLabelCompare<CompactLatticeArc>());

        // Wraps the rnnlm into FST. We re-create it for each lattice to prevent
        // memory usage increasing with time.
        RnnlmDeterministicFst rnnlm_fst(max_ngram_order, &rnnlm);

        // Composes lattice with language model.
        CompactLattice composed_clat;
        ComposeCompactLatticeDeterministic(clat, &rnnlm_fst, &composed_clat);

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
