// latbin/lattice-compose.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//           2022  Brno University of Technology

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
        "Composes lattices (in transducer form, as type Lattice).\n"
        "Depending on the command-line arguments, either composes\n"
        "lattices with lattices, or lattices with a single FST or\n"
        " multiple FSTs (whose weights are interpreted as \"graph weights\").\n"
        "\n"
        "Usage: lattice-compose [options] <lattice-rspecifier1> "
        "<lattice-rspecifier2|fst-rxfilename2|fst-rspecifier2> <lattice-wspecifier>\n"
        "If the 2nd arg is an rspecifier, it is interpreted by default as a table of\n"
        "lattices, or as a table of FSTs if you specify --compose-with-fst=true.\n";

    ParseOptions po(usage);

    bool write_compact = true;
    int32 num_states_cache = 50000;
    int32 phi_label = fst::kNoLabel; // == -1
    int32 rho_label = fst::kNoLabel; // == -1
    std::string compose_with_fst = "auto";

    po.Register("write-compact", &write_compact, "If true, write in normal (compact) form.");
    po.Register("phi-label", &phi_label, "If >0, the label on backoff arcs of the LM");
    po.Register("rho-label", &rho_label,
                "If >0, the label to forward lat1 paths not present in biasing graph fst2 "
                "(rho is input and output symbol on special arc in biasing graph fst2;"
                " rho is like phi (matches rest), but rho label is rewritten to the"
                " specific symbol from lat1)");
    po.Register("num-states-cache", &num_states_cache,
                "Number of states we cache when mapping LM FST to lattice type. "
                "More -> more memory but faster.");
    po.Register("compose-with-fst", &compose_with_fst,
                "(true|false|auto) For auto arg2 is: rspecifier=lats, rxfilename=fst "
                "(old behavior), for true/false rspecifier is fst/lattice.");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(phi_label > 0 || phi_label == fst::kNoLabel); // e.g. 0 not allowed.
    KALDI_ASSERT(rho_label > 0 || rho_label == fst::kNoLabel); // e.g. 0 not allowed.
    if (phi_label > 0 && rho_label > 0) {
      KALDI_ERR << "You cannot set both 'phi_label' and 'rho_label' at the same time.";
    }

    { // convert 'compose_with_fst' to lowercase to support: true, True, TRUE
      std::string& str(compose_with_fst);
      std::transform(str.begin(), str.end(), str.begin(), (int(*)(int))std::tolower); // lc
    }
    if (compose_with_fst != "auto" && compose_with_fst != "true" &&
                                                 compose_with_fst != "false") {
      KALDI_ERR << "Unkown 'compose_with_fst' value : " << compose_with_fst
                << " , values are (auto|true|false)";
    }

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

    bool arg2_is_rxfilename = (ClassifyRspecifier(arg2, NULL, NULL) == kNoRspecifier);

    if (arg2_is_rxfilename && (compose_with_fst == "auto" || compose_with_fst == "true")) {
      /**
       * arg2 is rxfilename that contains a single fst
       * - compose arg1 lattices with single fst in arg2
       */
      std::string fst_rxfilename = arg2;
      VectorFst<StdArc>* fst2 = fst::ReadFstKaldi(fst_rxfilename);

      // Make sure fst2 is sorted on ilabel
      if (fst2->Properties(fst::kILabelSorted, true) == 0) {
        fst::ILabelCompare<StdArc> ilabel_comp;
        ArcSort(fst2, ilabel_comp);
      }

      if (phi_label > 0)
        PropagateFinal(phi_label, fst2);

      // mapped_fst2 is fst2 interpreted using the LatticeWeight semiring,
      // with all the cost on the first member of the pair (since we're
      // assuming it's a graph weight).
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
        } else if (rho_label > 0) {
          RhoCompose(lat1, mapped_fst2, rho_label, &composed_lat);
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

    } else if (arg2_is_rxfilename && compose_with_fst == "false") {
      /**
       * arg2 is rxfilename that contains a single lattice
       * - would it make sense to do this? Not implementing...
       */
      KALDI_ERR << "Unimplemented...";

    } else if (!arg2_is_rxfilename &&
                  (compose_with_fst == "auto" || compose_with_fst == "false")) {
      /**
       * arg2 is rspecifier that contains a table of lattices
       * - composing arg1 lattices with arg2 lattices
       */
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

        Lattice composed_lat;
        // Btw, can the lat2 lattice contin phi/rho symbols ?
        if (phi_label > 0) {
          PropagateFinal(phi_label, &lat2);
          PhiCompose(lat1, lat2, phi_label, &composed_lat);
        } else if (rho_label > 0) {
          RhoCompose(lat1, lat2, rho_label, &composed_lat);
        } else {
          Compose(lat1, lat2, &composed_lat);
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

    } else if (!arg2_is_rxfilename && compose_with_fst == "true") {
      /**
       * arg2 is rspecifier that contains a table of fsts
       * - composing arg1 lattices with arg2 fsts
       */
      std::string fst_rspecifier2 = arg2;
      RandomAccessTableReader<fst::VectorFstHolder> fst_reader2(fst_rspecifier2);

      for (; !lattice_reader1.Done(); lattice_reader1.Next()) {
        std::string key = lattice_reader1.Key();
        KALDI_VLOG(1) << "Processing lattice for key " << key;
        Lattice lat1 = lattice_reader1.Value();
        lattice_reader1.FreeCurrent();

        if (!fst_reader2.HasKey(key)) {
          KALDI_WARN << "Not producing output for utterance " << key
                     << " because not present in second table.";
          n_fail++;
          continue;
        }

        VectorFst<StdArc> fst2 = fst_reader2.Value(key);
        // Make sure fst2 is sorted on ilabel
        if (fst2.Properties(fst::kILabelSorted, true) == 0) {
          fst::ILabelCompare<StdArc> ilabel_comp;
          fst::ArcSort(&fst2, ilabel_comp);
        }

        // for composing with LM-fsts, it makes all fst2 states final
        if (phi_label > 0)
          PropagateFinal(phi_label, &fst2);

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
        if (phi_label > 0) {
          PhiCompose(lat1, mapped_fst2, phi_label, &composed_lat);
        } else if (rho_label > 0) {
          RhoCompose(lat1, mapped_fst2, rho_label, &composed_lat);
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
    } else {
      /**
       * none of the 'if-else-if' applied...
       */
      KALDI_ERR << "You should never reach here...";
    }

    KALDI_LOG << "Done " << n_done << " lattices; failed for "
              << n_fail;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
