// latbin/lattice-arcgraph.cc

// Copyright 2012 BUT (author: Mirko Hannemann)

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

#include "fst/fstlib.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "hmm/transition-model.h"

typedef fst::StdArc::StateId StateId;
typedef fst::StdArc::Weight Weight;
typedef fst::StdArc::Label Label;

/* does the same as ConvertFstToLattice, but
 we encode HCLG state and transition into the output labels
 (which is what we later use as HCLG arc graph)
 and we replace transition-ids with transition states to make it independent
 of whether self-loops occur before or after the normal transitions,
 which is necessary, since the order was time reversed */
void ConvertFstToArcLattice(fst::StdVectorFst *net, kaldi::Lattice *lat,
                            std::vector<std::pair<int32,int32> > *arc_map,
                            kaldi::TransitionModel *tmodel, bool keep_weights) {
  int32 num_arcs = 0; // count to reserve the size of arc_map
  for(int32 i = 0; i < net->NumStates(); i++) {
    lat->AddState();
    num_arcs += net->NumArcs(i);
  }
  arc_map->reserve(num_arcs);
  lat->SetStart(net->Start());
  for(fst::StateIterator<fst::StdVectorFst> siter(*net); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    Weight w_final = net->Final(s);
    if (w_final != Weight::Zero()) { // final state
      kaldi::LatticeWeight new_weight(kaldi::LatticeWeight::One());
      if (keep_weights) new_weight.SetValue1(w_final.Value());
      lat->SetFinal(s, new_weight);
    }
    kaldi::int32 arc_id = 0;
    for (fst::MutableArcIterator<fst::StdVectorFst> aiter(net, s); !aiter.Done(); aiter.Next()) {
      fst::StdArc arc = aiter.Value();
      kaldi::LatticeWeight new_weight(kaldi::LatticeWeight::One());
      if (keep_weights) new_weight.SetValue1(arc.weight.Value());
      Label ilabel = arc.ilabel; // transition-id
      // transition state is independent of self-loop order
      if (ilabel > 0) ilabel = tmodel->TransitionIdToTransitionState(ilabel);
      arc_map->push_back(std::make_pair(s, arc_id));
      Label olabel = arc_map->size(); // new labels encode the unique index
      kaldi::LatticeArc new_arc(ilabel, olabel, new_weight, arc.nextstate);
      lat->AddArc(s, new_arc);
      arc_id ++;
    }
  }
}

/* Discards the ilabel on an arc, and replaces the (ilabel, olabel) with a pair
   obtained from the "arc_map" vector, indexed with the original olabel.  At
   input the (ilabel, olabel) of the graph are (transition-ids, indexes into
   arc_map), and when this function is done they are (states in HCLG,
   indexes into the lists of arcs leaving those states).  */
void DecodeGraphSymbols(const std::vector<std::pair<int32,int32> > &arc_map,
                        fst::StdVectorFst *net) {
  // maps symbols back state/arc pairs
  for(fst::StateIterator<fst::StdVectorFst> siter(*net);
      !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    for (fst::MutableArcIterator<fst::StdVectorFst> aiter(net, s);
         !aiter.Done(); aiter.Next()) {
      fst::StdArc arc = aiter.Value();
      Label ilabel = arc_map[arc.ilabel-1].first; // state
      Label olabel = arc_map[arc.ilabel-1].second; // arc
      fst::StdArc new_arc(ilabel, olabel, arc.weight, arc.nextstate);
      aiter.SetValue(new_arc);
    }
  }
}

void MapTransitionIdsToTransitionStates(kaldi::CompactLattice *lat,
                            kaldi::TransitionModel *tmodel, bool keep_weights) {
  // maps transition-ids to transition states and removes weights
  for(fst::StateIterator<kaldi::CompactLattice> siter(*lat);
      !siter.Done(); siter.Next()) {
    StateId s = siter.Value();

    kaldi::CompactLatticeWeight w_final = lat->Final(s);
    if (w_final.Weight() != kaldi::LatticeWeight::Zero()) { // final state
      std::vector<kaldi::int32> syms = w_final.String();
      // map all transition-ids to transition-states
      for(std::vector<kaldi::int32>::iterator it = syms.begin();
          it != syms.end(); ++it) {
        *it = tmodel->TransitionIdToTransitionState(*it);
      }
      kaldi::LatticeWeight new_w(w_final.Weight());
      if (!keep_weights) new_w = kaldi::LatticeWeight::One();
      kaldi::CompactLatticeWeight newwgt(new_w, syms); 
      lat->SetFinal(s, newwgt);
    }

    // go through all states
    for(fst::MutableArcIterator<kaldi::CompactLattice> aiter(lat, s);
        !aiter.Done(); aiter.Next()) {
      kaldi::CompactLatticeArc arc = aiter.Value();
      kaldi::CompactLatticeWeight w = arc.weight;
      std::vector<kaldi::int32> syms = w.String();

      // map all transition-ids to transition-states
      for(std::vector<kaldi::int32>::iterator it = syms.begin();
          it != syms.end(); ++it) {
        *it = tmodel->TransitionIdToTransitionState(*it);
      }
      kaldi::LatticeWeight new_w(w.Weight());
      if (!keep_weights) new_w = kaldi::LatticeWeight::One();
      kaldi::CompactLatticeWeight newwgt(new_w, syms); 
      arc.weight = newwgt;
      aiter.SetValue(arc);
    }
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    std::string lattice_wspecifier;
    std::string fst_out_filename;

    const char *usage =
        "Compose decoding graph with given lattices to obtain active-arc lattices.\n"
        "Usage: lattice-arcgraph [options] <model-file> "
        "<decoding-graph-fst> <lattice-rspecifier> <arcs-wspecifier>\n"
        " e.g.: lattice-arcgraph final.mdl HCLG.fst ark:in.lats ark:out.arcs\n";

    ParseOptions po(usage);
    po.Register("write-lattices", &lattice_wspecifier,
      "Write intermediate lattices to archive.");

    bool reverse = true;
    po.Register("reverse", &reverse, "Reverse input lattices in time");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        model_filename = po.GetArg(1),
        fst_in_filename = po.GetArg(2),
        lats_rspecifier = po.GetArg(3),
        arcs_wspecifier = po.GetArg(4);

    // options for lattice determinization
    fst::DeterminizeLatticeOptions lat_opts;
    lat_opts.max_mem = 200000000; // 200 MB
    lat_opts.max_loop = 500000;
    lat_opts.delta = fst::kDelta;

    // load transition model as begin of model file
    TransitionModel trans_model;
    ReadKaldiObject(model_filename, &trans_model);

    // read decoding graph and convert to transition-state acceptor lattice
    fst::StdVectorFst *hclg = fst::ReadFstKaldi(fst_in_filename);
    Lattice graph;
    // encode HCLG state and transition into the output labels
    // and replace transition-ids with transition states
    std::vector<std::pair<int32,int32> > arc_map; // maps state/arc to symbol-id
    ConvertFstToArcLattice(hclg, &graph, &arc_map, &trans_model, true); // keep weights
    fst::ArcSort(&graph, fst::ILabelCompare<kaldi::LatticeArc>());

    TableWriter<fst::VectorFstHolder> arcs_writer(arcs_wspecifier);
    LatticeWriter lat_writer;
    if (lattice_wspecifier != "") lat_writer.Open(lattice_wspecifier);

    // convert all lattices and compose with acceptor HCLG
    int32 n_done = 0, n_error = 0;
    kaldi::SequentialCompactLatticeReader lat_reader(lats_rspecifier);
    for (; !lat_reader.Done(); lat_reader.Next()) {
      std::string key = lat_reader.Key();
      CompactLattice lat = lat_reader.Value();
      lat_reader.FreeCurrent();

      // map transition-ids to self-loop independent and set weights to zero
      MapTransitionIdsToTransitionStates(&lat, &trans_model, false);

      // convert from CompactLattice to Lattice
      Lattice lat_mapped;
      ConvertLattice(lat, &lat_mapped);
      fst::Project(&lat_mapped, fst::PROJECT_INPUT); // keep only transition states

      Lattice lat_composed;
      if(reverse) {
        // reverse lattice in time
        Lattice lat_reverse;
        fst::Reverse(lat_mapped, &lat_reverse);
        RemoveEpsLocal(&lat_reverse);
        // compose
        KALDI_LOG << "compose with lattice " << key;
        Compose(lat_reverse, graph, &lat_composed); // composed FST contains HCLG arcs
      } else {
        KALDI_LOG << "compose with lattice " << key;
        Compose(lat_mapped, graph, &lat_composed); // composed FST contains HCLG arcs
      }
      if (lattice_wspecifier != "") lat_writer.Write(key, lat_composed);

      CompactLattice clat_determinized;
      if (DeterminizeLattice(lat_composed, &clat_determinized, lat_opts, NULL)) { 
        // now we can forget about the weights
        ScaleLattice(fst::LatticeScale(0.0, 0.0), &clat_determinized);
        Lattice lat_det;
        ConvertLattice(clat_determinized, &lat_det);

        // convert to StdFst, remove word labels
        fst::VectorFst<fst::StdArc> fst_det;
        fst::LatticeToStdMapper<BaseFloat> mapper;
        Map(lat_det, &fst_det, mapper);
        Project(&fst_det, fst::PROJECT_INPUT);
        ArcSort(&fst_det, fst::ILabelCompare<fst::StdArc>()); //to improve speed

        // determinize again on the arc_ids
        fst::VectorFst<fst::StdArc> fst_final;
        bool debug_location = false;
        DeterminizeStar(fst_det, &fst_final, fst::kDelta, &debug_location, -1);
        DecodeGraphSymbols(arc_map, &fst_final);
        ArcSort(&fst_final, fst::OLabelCompare<fst::StdArc>());
        // the decoder expects the arc numbers to be sorted
        arcs_writer.Write(key, fst_final);

        KALDI_LOG << key << " finished";
        n_done++;
      } else {
        n_error++; // will have already printed warning.
      }
    }
    if (lattice_wspecifier != "") lat_writer.Close();

    KALDI_LOG << "Done converting " << n_done << " lattices, failed: " << n_error;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
