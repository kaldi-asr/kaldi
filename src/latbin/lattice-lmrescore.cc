//latbin/lattice-lmrescore.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University (author: Daniel Povey)

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
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

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
        "Add lm_scale * [cost of best path through LM FST] to graph-cost of\n"
        "paths through lattice.  Does this by composing with LM FST, then\n"
        "lattice-determinizing (it has to negate weights first if lm_scale<0)\n"
        "Usage: lattice-lmrescore [options] <lattice-rspecifier> <lm-fst-in> <lattice-wspecifier>\n"
        " e.g.: lattice-lmrescore --lm-scale=-1.0 ark:in.lats 'fstproject --project_output=true data/lang/G.fst|' ark:out.lats\n";

    ParseOptions po(usage);
    bool write_compact = true;
    BaseFloat lm_scale = 1.0;
    int32 num_states_cache = 50000;

    po.Register("write-compact", &write_compact, "If true, write in normal (compact) form.");
    po.Register("lm-scale", &lm_scale, "Scaling factor for language model costs; frequently 1.0 or -1.0");
    po.Register("num-states-cache", &num_states_cache,
                "Number of states we cache when mapping LM FST to lattice type. "
                "More -> more memory but faster.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);



    VectorFst<StdArc> *std_lm_fst = ReadFstKaldi(fst_rxfilename);
    if (std_lm_fst->Properties(fst::kILabelSorted, true) == 0) {
      // Make sure LM is sorted on ilabel.
      fst::ILabelCompare<StdArc> ilabel_comp;
      fst::ArcSort(std_lm_fst, ilabel_comp);
    }

    // mapped_fst is the LM fst interpreted using the LatticeWeight semiring,
    // with all the cost on the first member of the pair (since it's a graph
    // weight).
    fst::CacheOptions cache_opts(true, num_states_cache);
    fst::MapFstOptions mapfst_opts(cache_opts);
    fst::StdToLatticeMapper<BaseFloat> mapper;
    fst::MapFst<StdArc, LatticeArc, fst::StdToLatticeMapper<BaseFloat> >
        lm_fst(*std_lm_fst, mapper, mapfst_opts);
    delete std_lm_fst;

    // The next fifteen or so lines are a kind of optimization and
    // can be ignored if you just want to understand what is going on.
    // Change the options for TableCompose to match the input
    // (because it's the arcs of the LM FST we want to do lookup
    // on).
    fst::TableComposeOptions compose_opts(fst::TableMatcherOptions(),
                                          true, fst::SEQUENCE_FILTER,
                                          fst::MATCH_INPUT);

    // The following is an optimization for the TableCompose
    // composition: it stores certain tables that enable fast
    // lookup of arcs during composition.
    fst::TableComposeCache<fst::Fst<LatticeArc> > lm_compose_cache(compose_opts);

    // Read as regular lattice-- this is the form we need it in for efficient
    // composition and determinization.
    SequentialLatticeReader lattice_reader(lats_rspecifier);

    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    
    if (write_compact)
      compact_lattice_writer.Open(lats_wspecifier);
    else
      lattice_writer.Open(lats_wspecifier);

    int32 n_done = 0, n_fail = 0;

    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      Lattice lat = lattice_reader.Value();
      lattice_reader.FreeCurrent();

      // Compute a map from each (t, tid) to (sum_of_acoustic_scores, count)
      unordered_map<std::pair<int32,int32>, std::pair<BaseFloat, int32>,
                                          PairHasher<int32> > acoustic_scores;
      if (!write_compact)
        ComputeAcousticScoresMap(lat, &acoustic_scores);

      if (lm_scale != 0.0) {
        // Only need to modify it if LM scale nonzero.
        // Before composing with the LM FST, we scale the lattice weights
        // by the inverse of "lm_scale".  We'll later scale by "lm_scale".
        // We do it this way so we can determinize and it will give the
        // right effect (taking the "best path" through the LM) regardless
        // of the sign of lm_scale.
        fst::ScaleLattice(fst::GraphLatticeScale(1.0 / lm_scale), &lat);
        ArcSort(&lat, fst::OLabelCompare<LatticeArc>());

        Lattice composed_lat;
        // Could just do, more simply: Compose(lat, lm_fst, &composed_lat);
        // and not have lm_compose_cache at all.
        // The command below is faster, though; it's constant not
        // logarithmic in vocab size.
        TableCompose(lat, lm_fst, &composed_lat, &lm_compose_cache);

        Invert(&composed_lat); // make it so word labels are on the input.

        CompactLattice determinized_clat;
        DeterminizeLattice(composed_lat, &determinized_clat);
        fst::ScaleLattice(fst::GraphLatticeScale(lm_scale), &determinized_clat);
        if (determinized_clat.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty lattice for utterance " << key
              << " (incompatible LM?)";
          n_fail++;
        } else {
          if (write_compact) {
            compact_lattice_writer.Write(key, determinized_clat);
          } else {
            // Replace each arc (t, tid) with the averaged acoustic score from
            // the computed map
            Lattice out_lat;
            ConvertLattice(determinized_clat, &out_lat);
            ReplaceAcousticScoresFromMap(acoustic_scores, &out_lat);
            lattice_writer.Write(key, out_lat);
          }
          n_done++;
        }
      } else {
        // Zero scale so nothing to do.
        n_done++;

        if (write_compact) {
          CompactLattice compact_lat;
          ConvertLattice(lat, &compact_lat);
          compact_lattice_writer.Write(key, compact_lat);
        } else {
          lattice_writer.Write(key, lat);
        }
      }
    }

    KALDI_LOG << "Done " << n_done << " lattices, failed for " << n_fail;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
