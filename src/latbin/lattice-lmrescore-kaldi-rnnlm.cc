// latbin/lattice-lmrescore-kaldi-rnnlm.cc

// Copyright 2017 Johns Hopkins University (author: Daniel Povey)
//           2017 Hainan Xu
//           2017 Yiming Wang

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
#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "util/common-utils.h"
#include "nnet3/nnet-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Rescores lattice with kaldi-rnnlm. This script is called from \n"
        "scripts/rnnlm/lmrescore.sh. An example for rescoring \n"
        "lattices is at egs/swbd/s5c/local/rnnlm/run_lstm.sh \n"
        "\n"
        "Usage: lattice-lmrescore-kaldi-rnnlm [options] \\\n"
        "             <embedding-file> <raw-rnnlm-rxfilename> \\\n"
        "             <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-lmrescore-kaldi-rnnlm --lm-scale=-1.0 \\\n"
        "              word_embedding.mat \\\n"
        "              --bos-symbol=1 --eos-symbol=2 \\\n"
        "              final.raw ark:in.lats ark:out.lats\n";

    ParseOptions po(usage);
    rnnlm::RnnlmComputeStateComputationOptions opts;

    int32 max_ngram_order = 3;
    BaseFloat lm_scale = 1.0;

    po.Register("lm-scale", &lm_scale, "Scaling factor for language model "
                "costs");
    po.Register("max-ngram-order", &max_ngram_order,
        "If positive, allow RNNLM histories longer than this to be identified "
        "with each other for rescoring purposes (an approximation that "
        "saves time and reduces output lattice size).");
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    if (opts.bos_index == -1 || opts.eos_index == -1) {
      KALDI_ERR << "You must set --bos-symbol and --eos-symbol options";
    }

    std::string word_embedding_rxfilename = po.GetArg(1),
                rnnlm_rxfilename = po.GetArg(2),
                lats_rspecifier = po.GetArg(3),
                lats_wspecifier = po.GetArg(4);

    kaldi::nnet3::Nnet rnnlm;
    ReadKaldiObject(rnnlm_rxfilename, &rnnlm);

    KALDI_ASSERT(IsSimpleNnet(rnnlm));

    CuMatrix<BaseFloat> word_embedding_mat;
    ReadKaldiObject(word_embedding_rxfilename, &word_embedding_mat);

    const rnnlm::RnnlmComputeStateInfo info(opts, rnnlm, word_embedding_mat);

    // Reads and writes as compact lattice.
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_done = 0, n_fail = 0;

    rnnlm::KaldiRnnlmDeterministicFst rnnlm_fst(max_ngram_order, info);

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
      rnnlm_fst.Clear();
    }

    KALDI_LOG << "Done " << n_done << " lattices, failed for " << n_fail;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
