// latbin/lattice-lmrescore-kaldi-rnnlm.cc

// Copyright 2017 Johns Hopkins University (author: Daniel Povey)
//                Hainan Xu
//                Yiming Wang

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
        "Rescores lattice with rnnlm. The LM will be wrapped into the\n"
        "DeterministicOnDemandFst interface and the rescoring is done by\n"
        "composing with the wrapped LM using a special type of composition\n"
        "algorithm. Determinization will be applied on the composed lattice.\n"
        "\n"
        "Usage: lattice-lmrescore-kaldi-rnnlm [options] <embedding-file> <rnnlm-wordlist> \\\n"
        "             <word-symbol-table-rxfilename> <lattice-rspecifier> \\\n"
        "             <raw-rnnlm-rxfilename> <lattice-wspecifier>\n"
        " e.g.: lattice-lmrescore-kaldi-rnnlm --lm-scale=-1.0 word_embedding.mat \\\n"
        "       rnn_words.txt fst_words.txt ark:in.lats rnnlm ark:out.lats\n";

    ParseOptions po(usage);
    int32 max_ngram_order = 3;
    BaseFloat lm_scale = 1.0;

    po.Register("lm-scale", &lm_scale, "Scaling factor for language model "
                "costs; frequently 1.0 or -1.0");
    po.Register("max-ngram-order", &max_ngram_order, "If positive, limit the "
                "rnnlm context to the given number, -1 means we are not going "
                "to limit it.");

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier, rnn_wordlist, word_embedding_rxfilename,
        word_symbols_rxfilename, rnnlm_rxfilename, lats_wspecifier;

    word_embedding_rxfilename = po.GetArg(1);
    rnn_wordlist = po.GetArg(2);
    word_symbols_rxfilename = po.GetArg(3);
    lats_rspecifier = po.GetArg(4);
    rnnlm_rxfilename = po.GetArg(5);
    lats_wspecifier = po.GetArg(6);

    // Reads the language model.
    kaldi::nnet3::Nnet rnnlm;
    ReadKaldiObject(rnnlm_rxfilename, &rnnlm);

    if (!IsSimpleNnet(rnnlm))
      KALDI_ERR << "Input RNNLM in " << rnnlm_rxfilename
                << " is not the type of neural net we were looking for; "
          "failed IsSimpleNnet().";

    CuMatrix<BaseFloat> word_embedding_mat;
    ReadKaldiObject(word_embedding_rxfilename, &word_embedding_mat);

    const nnet3::DecodableRnnlmSimpleLoopedComputationOptions opts;
    const nnet3::DecodableRnnlmSimpleLoopedInfo info(opts, rnnlm, word_embedding_mat);

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
        fst::ScaleLattice(fst::GraphLatticeScale(1.0 / lm_scale), &clat);
        ArcSort(&clat, fst::OLabelCompare<CompactLatticeArc>());

        // Wraps the rnnlm into FST. We re-create it for each lattice to prevent
        // memory usage increasing with time.
        nnet3::KaldiRnnlmDeterministicFst rnnlm_fst(max_ngram_order,
                                                    rnn_wordlist,
                                                    word_symbols_rxfilename,
                                                    info);

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
