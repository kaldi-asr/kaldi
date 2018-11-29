// latbin/lattice-compose.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University

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
        "Composes lattices (in transducer form, as type Lattice).  Depending\n"
        "on the command-line arguments, either composes lattices with lattices,\n"
        "or lattices with FSTs (rspecifiers are assumed to be lattices, and\n"
        "rxfilenames are assumed to be FSTs, which have their weights interpreted\n"
        "as \"graph weights\" when converted into the Lattice format.\n"
        "\n"
        "Usage: lattice-compose [options] lattice-rspecifier1 "
        "(lattice-rspecifier2|fst-rxfilename2) lattice-wspecifier\n"
        " e.g.: lattice-compose ark:1.lats ark:2.lats ark:composed.lats\n"
        " or: lattice-compose ark:1.lats G.fst ark:composed.lats\n";

    ParseOptions po(usage);

    bool write_compact = true;
    int32 num_states_cache = 50000;
    int32 phi_label = fst::kNoLabel; // == -1
    po.Register("write-compact", &write_compact, "If true, write in normal (compact) form.");
    po.Register("phi-label", &phi_label, "If >0, the label on backoff arcs of the LM");
    po.Register("num-states-cache", &num_states_cache,
                "Number of states we cache when mapping LM FST to lattice type. "
                "More -> more memory but faster.");
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

    if (write_compact)
      compact_lattice_writer.Open(lats_wspecifier);
    else
      lattice_writer.Open(lats_wspecifier);

    if (ClassifyRspecifier(arg2, NULL, NULL) == kNoRspecifier) {
      std::string fst_rxfilename = arg2;
      VectorFst<StdArc> *fst2 = fst::ReadFstKaldi(fst_rxfilename);
      // mapped_fst2 is fst2 interpreted using the LatticeWeight semiring,
      // with all the cost on the first member of the pair (since we're
      // assuming it's a graph weight).
      if (fst2->Properties(fst::kILabelSorted, true) == 0) {
        // Make sure fst2 is sorted on ilabel.
        fst::ILabelCompare<StdArc> ilabel_comp;
        ArcSort(fst2, ilabel_comp);
      }
      if (phi_label > 0)
        PropagateFinal(phi_label, fst2);

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
      delete fst2;
    } else {
      std::string lats_rspecifier2 = arg2;
      // This is the case similar to lattice-interp.cc, where we
      // read in another set of lattices and compose them.  But in this
      // case we don't do any projection; we assume that the user has already
      // done this (e.g. with lattice-project).
      RandomAccessLatticeReader lattice_reader2(lats_rspecifier2);
      for (; !lattice_reader1.Done(); lattice_reader1.Next()) {
        std::string key = lattice_reader1.Key();
        KALDI_VLOG(1) << "Processing lattice for key " << key;
        Lattice lat1 = lattice_reader1.Value();
        lattice_reader1.FreeCurrent();
        if (!lattice_reader2.HasKey(key)) {
          KALDI_WARN << "Not producing output for utterance " << key
                     << " because not present in second table.";
          n_fail++;
          continue;
        }
        Lattice lat2 = lattice_reader2.Value(key);
        // Make sure that either lat2 is ilabel sorted
        // or lat1 is olabel sorted, to ensure that
        // composition will work.
        if (lat2.Properties(fst::kILabelSorted, true) == 0
            && lat1.Properties(fst::kOLabelSorted, true) == 0) {
          // arbitrarily choose to sort lat2 rather than lat1.
          fst::ILabelCompare<LatticeArc> ilabel_comp;
          fst::ArcSort(&lat2, ilabel_comp);
        }

        Lattice lat_out;
        if (phi_label > 0) {
          PropagateFinal(phi_label, &lat2);
          PhiCompose(lat1, lat2, phi_label, &lat_out);
        } else {
          Compose(lat1, lat2, &lat_out);
        }
        if (lat_out.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty lattice for utterance " << key << " (incompatible LM?)";
          n_fail++;
        } else {
          if (write_compact) {
            CompactLattice clat_out;
            ConvertLattice(lat_out, &clat_out);
            compact_lattice_writer.Write(key, clat_out);
          } else {
            lattice_writer.Write(key, lat_out);
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
