// latbin/lattice-lmrescore-pruned.cc

// Copyright      2017  Johns Hopkins University (author: Daniel Povey)

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
#include "fstext/kaldi-fst-io.h"
#include "lm/const-arpa-lm.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "lat/compose-lattice-pruned.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;
    using fst::ReadFstKaldi;

    const char *usage =
        "This program can be used to subtract scores from one language model and\n"
        "add scores from another one.  It uses an efficient rescoring algorithm that\n"
        "avoids exploring the entire composed lattice.  The first (negative-weight)\n"
        "language model is expected to be an FST, e.g. G.fst; the second one can\n"
        "either be in FST or const-arpa format.  Any FST-format language models will\n"
        "be projected on their output by this program, making it unnecessary for the\n"
        "caller to remove disambiguation symbols.\n"
        "\n"
        "Usage: lattice-lmrescore-pruned [options] <lm-to-subtract> <lm-to-add> <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-lmrescore-pruned --acoustic-scale=0.1 \\\n"
        "      data/lang/G.fst data/lang_fg/G.fst ark:in.lats ark:out.lats\n"
        " or: lattice-lmrescore-pruned --acoustic-scale=0.1 --add-const-arpa=true\\\n"
        "      data/lang/G.fst data/lang_fg/G.carpa ark:in.lats ark:out.lats\n";

    ParseOptions po(usage);

    // the options for the composition include --lattice-compose-beam,
    // --max-arcs and --growth-ratio.
    ComposeLatticePrunedOptions compose_opts;
    BaseFloat lm_scale = 1.0;
    BaseFloat acoustic_scale = 1.0;
    bool add_const_arpa = false;

    po.Register("lm-scale", &lm_scale, "Scaling factor for <lm-to-add>; its negative "
                "will be applied to <lm-to-subtract>.");
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic "
                "probabilities (e.g. 0.1 for non-chain systems); important because "
                "of its effect on pruning.");
    po.Register("add-const-arpa", &add_const_arpa, "If true, <lm-to-add> is expected"
                "to be in const-arpa format; if false it's expected to be in FST"
                "format.");


    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string lm_to_subtract_rxfilename = po.GetArg(1),
        lm_to_add_rxfilename = po.GetArg(2),
        lats_rspecifier = po.GetArg(3),
        lats_wspecifier = po.GetArg(4);

    KALDI_LOG << "Reading LMs...";
    VectorFst<StdArc> *lm_to_subtract_fst = fst::ReadAndPrepareLmFst(
        lm_to_subtract_rxfilename);
    VectorFst<StdArc> *lm_to_add_fst = NULL;
    ConstArpaLm const_arpa;
    if (add_const_arpa) {
      ReadKaldiObject(lm_to_add_rxfilename, &const_arpa);
    } else {
      lm_to_add_fst = fst::ReadAndPrepareLmFst(lm_to_add_rxfilename);
    }
    fst::BackoffDeterministicOnDemandFst<StdArc> lm_to_subtract_det_backoff(
        *lm_to_subtract_fst);
    fst::ScaleDeterministicOnDemandFst lm_to_subtract_det_scale(
        -lm_scale, &lm_to_subtract_det_backoff);


    fst::DeterministicOnDemandFst<StdArc> *lm_to_add_orig = NULL,
        *lm_to_add = NULL;
    if (add_const_arpa) {
      lm_to_add = new ConstArpaLmDeterministicFst(const_arpa);
    } else {
      lm_to_add = new fst::BackoffDeterministicOnDemandFst<StdArc>(
          *lm_to_add_fst);
    }
    if (lm_scale != 1.0) {
      lm_to_add_orig = lm_to_add;
      lm_to_add = new fst::ScaleDeterministicOnDemandFst(lm_scale,
                                                         lm_to_add_orig);
    }

    KALDI_LOG << "Done.";

    // We read and write as CompactLattice.
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);

    // Write as compact lattice.
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 num_done = 0, num_err = 0;

    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice &clat = clat_reader.Value();

      if (acoustic_scale != 1.0) {
        fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &clat);
      }
      TopSortCompactLatticeIfNeeded(&clat);

      // To avoid memory gradually increasing with time, we reconstruct the
      // composed-LM FST for each lattice we process.
      //   It shouldn't make a difference in which order we provide the
      // arguments to the composition; either way should work.  They are both
      // acceptors so the result is the same either way.
      fst::ComposeDeterministicOnDemandFst<StdArc> combined_lms(
          &lm_to_subtract_det_scale, lm_to_add);

      CompactLattice composed_clat;
      ComposeCompactLatticePruned(compose_opts,
                                  clat,
                                  &combined_lms,
                                  &composed_clat);

      if (composed_clat.NumStates() == 0) {
        // Something went wrong.  A warning will already have been printed.
        num_err++;
      } else {
        if (acoustic_scale != 1.0) {
          if (acoustic_scale == 0.0)
            KALDI_ERR << "Acoustic scale cannot be zero.";
          fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale),
                            &composed_clat);
        }
        compact_lattice_writer.Write(key, composed_clat);
        num_done++;
      }
    }
    delete lm_to_subtract_fst;
    delete lm_to_add_fst;
    delete lm_to_add_orig;
    delete lm_to_add;

    KALDI_LOG << "Overall, succeeded for " << num_done
              << " lattices, failed for " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
