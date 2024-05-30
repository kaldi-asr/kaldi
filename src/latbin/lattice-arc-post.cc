// latbin/lattice-arc-post.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

namespace kaldi {

// This class computes and outputs
// the information about arc posteriors.

class ArcPosteriorComputer {
 public:
  // Note: 'clat' must be topologically sorted.
  ArcPosteriorComputer(const CompactLattice &clat,
                       BaseFloat min_post,
                       bool print_alignment,
                       const TransitionModel *trans_model = NULL):
      clat_(clat), min_post_(min_post), print_alignment_(print_alignment),
      trans_model_(trans_model) { }

  // returns the number of arc posteriors that it output.
  int32 OutputPosteriors(const std::string &utterance,
                         std::ostream &os) {
    int32 num_post = 0;
    if (!ComputeCompactLatticeAlphas(clat_, &alpha_))
      return num_post;
    if (!ComputeCompactLatticeBetas(clat_, &beta_))
      return num_post;

    CompactLatticeStateTimes(clat_, &state_times_);
    if (clat_.Start() < 0)
      return 0;
    double tot_like = beta_[clat_.Start()];

    int32 num_states = clat_.NumStates();
    for (int32 state = 0; state < num_states; state++) {
      for (fst::ArcIterator<CompactLattice> aiter(clat_, state);
           !aiter.Done(); aiter.Next()) {
        const CompactLatticeArc &arc = aiter.Value();
        double arc_loglike = -ConvertToCost(arc.weight) +
            alpha_[state] + beta_[arc.nextstate] - tot_like;
        KALDI_ASSERT(arc_loglike < 0.1 &&
                     "Bad arc posterior in forward-backward computation");
        if (arc_loglike > 0.0) arc_loglike = 0.0;
        int32 num_frames = arc.weight.String().size(),
            word = arc.ilabel;
        BaseFloat arc_post = exp(arc_loglike);
        if (arc_post <= min_post_) continue;
        os << utterance << '\t' << state_times_[state] << '\t' << num_frames
           << '\t' << arc_post << '\t' << word;
        if (print_alignment_) {
          os << '\t';
          const std::vector<int32> &ali = arc.weight.String();
          for (int32 frame = 0; frame < num_frames; frame++) {
            os << ali[frame];
            if (frame + 1 < num_frames) os << ',';
          }
        }
        if (trans_model_ != NULL) {
          // we want to print the phone sequence too.
          os << '\t';
          const std::vector<int32> &ali = arc.weight.String();
          bool first_phone = true;
          for (int32 frame = 0; frame < num_frames; frame++) {
            if (trans_model_->IsFinal(ali[frame])) {
              if (first_phone) first_phone = false;
              else os << ' ';
              os << trans_model_->TransitionIdToPhone(ali[frame]);
            }
          }
        }
        os << std::endl;
        num_post++;
      }
    }
    return num_post;
  }
 private:
  const CompactLattice &clat_;
  std::vector<double> alpha_;
  std::vector<double> beta_;
  std::vector<int32> state_times_;

  BaseFloat min_post_;
  bool print_alignment_;
  const TransitionModel *trans_model_;
};

}


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Print out information regarding posteriors of lattice arcs\n"
        "This program computes posteriors from a lattice and prints out\n"
        "information for each arc (the format is reminiscent of ctm, but\n"
        "contains information from multiple paths).  Each line is:\n"
        " <utterance-id> <start-frame> <num-frames> <posterior> <word> [<ali>] [<phone1> <phone2>...]\n"
        "for instance:\n"
        "2013a04-bk42\t104\t26\t0.95\t0\t11,242,242,242,71,894,894,62,63,63,63,63\t2 8 9\n"
        "where the --print-alignment option determines whether the alignments (i.e. the\n"
        "sequences of transition-ids) are printed, and the phones are printed only if the\n"
        "<model> is supplied on the command line.  Note, there are tabs between the major\n"
        "fields, but the phones are separated by spaces.\n"
        "Usage: lattice-arc-post [<model>] <lattices-rspecifier> <output-wxfilename>\n"
        "e.g.: lattice-arc-post --acoustic-scale=0.1 final.mdl 'ark:gunzip -c lat.1.gz|' post.txt\n"
        "You will probably want to word-align the lattices (e.g. lattice-align-words or\n"
        "lattice-align-words-lexicon) before this program, apply an acoustic scale either\n"
        "via the --acoustic-scale option or using lattice-scale.\n"
        "See also: lattice-post, lattice-to-ctm-conf, nbest-to-ctm\n";

    kaldi::BaseFloat acoustic_scale = 1.0, lm_scale = 1.0;
    kaldi::BaseFloat min_post = 0.0001;
    bool print_alignment = false;

    kaldi::ParseOptions po(usage);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale,
                "Scaling factor for \"graph costs\" (including LM costs)");
    po.Register("print-alignment", &print_alignment,
                "If true, print alignments (i.e. sequences of transition-ids) for each\n"
                "arc.");
    po.Register("min-post", &min_post,
                "Arc posteriors below this value will be pruned away");
    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    if (acoustic_scale == 0.0)
      KALDI_ERR << "Do not use a zero acoustic scale (cannot be inverted)";

    kaldi::TransitionModel trans_model;

    std::string lats_rspecifier, output_wxfilename;
    if (po.NumArgs() == 3) {
      ReadKaldiObject(po.GetArg(1), &trans_model);
      lats_rspecifier = po.GetArg(2);
      output_wxfilename = po.GetArg(3);
    } else {
      lats_rspecifier = po.GetArg(1);
      output_wxfilename = po.GetArg(2);
    }


    kaldi::Output output(output_wxfilename, false);

    // Read as regular lattice
    kaldi::SequentialCompactLatticeReader clat_reader(lats_rspecifier);

    int64 tot_post = 0;
    int32 num_lat_done = 0, num_lat_err = 0;

    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      kaldi::CompactLattice clat = clat_reader.Value();
      // FreeCurrent() is an optimization that prevents the lattice from being
      // copied unnecessarily (OpenFst does copy-on-write).
      clat_reader.FreeCurrent();
      fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);
      kaldi::TopSortCompactLatticeIfNeeded(&clat);

      kaldi::ArcPosteriorComputer computer(
          clat, min_post, print_alignment,
          (po.NumArgs() == 3 ? &trans_model : NULL));

      int32 num_post = computer.OutputPosteriors(key, output.Stream());
      if (num_post != 0) {
        num_lat_done++;
        tot_post += num_post;
      } else {
        num_lat_err++;
        KALDI_WARN << "No posterior printed for " << key;
      }
    }
    KALDI_LOG << "Printed posteriors for " << num_lat_done << " lattices ("
              << num_lat_err << " with errors); on average printed "
              << (tot_post / (num_lat_done == 0 ? 1 : num_lat_done))
              << " posteriors per lattice.";
    return (num_lat_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
