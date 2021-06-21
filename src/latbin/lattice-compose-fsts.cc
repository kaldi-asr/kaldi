// latbin/lattice-compose-fsts.cc

// Copyright 2020  Brno University of Technology; Microsoft Corporation

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
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Composes lattices (in transducer form, as type Lattice) with word-network FSTs.\n"
        "Either with a single FST from rxfilename or with per-utterance FSTs from rspecifier.\n"
        "The FST weights are interpreted as \"graph weights\" when converted into the Lattice format.\n"
        "\n"
        "Usage: lattice-compose-fsts [options] lattice-rspecifier1 "
        "(fst-rspecifier2|fst-rxfilename2) lattice-wspecifier\n"
        " e.g.: lattice-compose-fsts ark:1.lats ark:2.fsts ark:composed.lats\n"
        " or: lattice-compose-fsts ark:1.lats G.fst ark:composed.lats\n";

    ParseOptions po(usage);

    bool write_compact = true;
    int32 num_states_cache = 50000;
    int32 phi_label = fst::kNoLabel;  // == -1
    po.Register("write-compact", &write_compact,
                "If true, write in normal (compact) form.");
    po.Register("phi-label", &phi_label,
                "If >0, the label on backoff arcs of the LM");
    po.Register("num-states-cache", &num_states_cache,
                "Number of states we cache when mapping LM FST to lattice type."
                " More -> more memory but faster.");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(phi_label > 0 || phi_label == fst::kNoLabel); // e.g. 0 not allowed.

    std::string lats_rspecifier1 = po.GetArg(1),
        arg2 = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);
    int32 n_done = 0, n_fail = 0;

    SequentialLatticeReader lattice_reader1(lats_rspecifier1);

    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;

    if (write_compact) {
      compact_lattice_writer.Open(lats_wspecifier);
    } else {
      lattice_writer.Open(lats_wspecifier);
    }

    if (ClassifyRspecifier(arg2, NULL, NULL) == kNoRspecifier) {
      std::string fst_rxfilename = arg2;
      VectorFst<StdArc>* fst2 = fst::ReadFstKaldi(fst_rxfilename);
      // mapped_fst2 is fst2 interpreted using the LatticeWeight semiring,
      // with all the cost on the first member of the pair (since we're
      // assuming it's a graph weight).
      if (fst2->Properties(fst::kILabelSorted, true) == 0) {
        // Make sure fst2 is sorted on ilabel.
        fst::ILabelCompare<StdArc> ilabel_comp;
        ArcSort(fst2, ilabel_comp);
      }
      /* // THIS MAKES ALL STATES FINAL STATES! WHY?
      if (phi_label > 0)
        PropagateFinal(phi_label, fst2);
      */

      fst::CacheOptions cache_opts(true, num_states_cache);
      fst::MapFstOptions mapfst_opts(cache_opts);
      fst::StdToLatticeMapper<BaseFloat> mapper;
      fst::MapFst<StdArc, LatticeArc, fst::StdToLatticeMapper<BaseFloat> >
          mapped_fst2(*fst2, mapper, mapfst_opts);

      for (; !lattice_reader1.Done(); lattice_reader1.Next()) {
        std::string key = lattice_reader1.Key();
        KALDI_VLOG(1) << "Processing lattice for key " << key;
        Lattice lat1 = lattice_reader1.Value();
        ArcSort(&lat1, fst::OLabelCompare<LatticeArc>());
        Lattice composed_lat;
        if (phi_label > 0) {
          PhiCompose(lat1, mapped_fst2, phi_label, &composed_lat);
        } else {
          Compose(lat1, mapped_fst2, &composed_lat);
        }
        if (composed_lat.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty lattice for utterance " << key << " (incompatible LM?)";
          n_fail++;
        } else {
          if (write_compact) {
            CompactLattice clat;
            ConvertLattice(composed_lat, &clat);
            compact_lattice_writer.Write(key, clat);
          } else {
            lattice_writer.Write(key, composed_lat);
          }
          n_done++;
        }
      }
      delete fst2;
    } else {
      // Compose each utterance with its matching (by key) FST.
      std::string fst_rspecifier2 = arg2;
      RandomAccessTableReader<fst::VectorFstHolder> fst_reader2(fst_rspecifier2);

      for (; !lattice_reader1.Done(); lattice_reader1.Next()) {
        std::string key = lattice_reader1.Key();
        KALDI_VLOG(1) << "Processing lattice for key " << key;
        Lattice lat1 = lattice_reader1.Value();
        lattice_reader1.FreeCurrent();

        if (!fst_reader2.HasKey(key)) {
          KALDI_WARN << "Not producing output for utterance " << key
                     << " because it's not present in second table.";
          n_fail++;
          continue;
        }

        VectorFst<StdArc> fst2 = fst_reader2.Value(key);
        if (fst2.Properties(fst::kILabelSorted, true) == 0) {
          // Make sure fst2 is sorted on ilabel.
          fst::ILabelCompare<StdArc> ilabel_comp;
          fst::ArcSort(&fst2, ilabel_comp);
        }
        /* // THIS MAKES ALL STATES FINAL STATES! WHY?
        if (phi_label > 0)
          PropagateFinal(phi_label, &fst2);
        */

        // mapped_fst2 is fst2 interpreted using the LatticeWeight semiring,
        // with all the cost on the first member of the pair (since we're
        // assuming it's a graph weight).
        fst::CacheOptions cache_opts(true, num_states_cache);
        fst::MapFstOptions mapfst_opts(cache_opts);
        fst::StdToLatticeMapper<BaseFloat> mapper;
        fst::MapFst<StdArc, LatticeArc, fst::StdToLatticeMapper<BaseFloat> >
            mapped_fst2(fst2, mapper, mapfst_opts);

        // sort lat1 on olabel.
        ArcSort(&lat1, fst::OLabelCompare<LatticeArc>());

        Lattice composed_lat;
        if (phi_label > 0) PhiCompose(lat1, mapped_fst2, phi_label, &composed_lat);
        else Compose(lat1, mapped_fst2, &composed_lat);

        if (composed_lat.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty lattice for utterance " << key << " (incompatible LM?)";
          n_fail++;
        } else {
          if (write_compact) {
            CompactLattice clat;
            ConvertLattice(composed_lat, &clat);
            compact_lattice_writer.Write(key, clat);
          } else {
            lattice_writer.Write(key, composed_lat);
          }
          n_done++;
        }
      }
    }

    KALDI_LOG << "Done " << n_done << " lattices; failed for "
              << n_fail;

    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
