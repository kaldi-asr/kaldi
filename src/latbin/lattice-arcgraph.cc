// latbin/lattice-arcgraph.cc

// Copyright 2012 BUT (author: Mirko Hannemann)

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

//#ifdef _MSC_VER
//#include <unordered_set>
//#else
//#include <tr1/unordered_set>
//#endif

#include "fst/fstlib.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

typedef fst::StdArc::StateId StateId;
typedef fst::StdArc::Weight Weight;
typedef fst::StdArc::Label Label;
//typedef std::tr1::unordered_set<StateId> StateSet;
//typedef std::tr1::unordered_set<int32> IdSet;


void ConvertFstToArcLattice(fst::StdVectorFst *net, kaldi::Lattice *lat,
                            std::vector<std::pair<int32,int32> > *arc_map,
                            kaldi::TransitionModel *tmodel, bool keep_weights) {
  // same as ConvertFstToLattice, but
  // encodes the arc and transition in the transducer
  // replaces transition-ids with transition states (self loop order)

  arc_map->reserve(net->NumStates()*4); // average fan-out is maybe 3
  for(int32 i = 0; i < net->NumStates(); i++) lat->AddState();
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
      //Label olabel = arc.olabel; // word labels 
      //Label olabel = s;
      //new_weight.SetValue2(float(arc_id)); // hack that assumes that composed weight will be zero
      arc_map->push_back(std::make_pair(s, arc_id));
      Label olabel = arc_map->size();
      kaldi::LatticeArc new_arc(ilabel, olabel, new_weight, arc.nextstate);
      lat->AddArc(s, new_arc);
      arc_id ++;
    }
  }
}

void DecodeGraphSymbols(fst::StdVectorFst *net,
                        std::vector<std::pair<int32,int32> > *arc_map) {
  // maps symbols back state/arc pairs
  for(fst::StateIterator<fst::StdVectorFst> siter(*net);
      !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    for (fst::MutableArcIterator<fst::StdVectorFst> aiter(net, s);
         !aiter.Done(); aiter.Next()) {
      fst::StdArc arc = aiter.Value();
      Label ilabel = (*arc_map)[arc.ilabel-1].first; // state
      Label olabel = (*arc_map)[arc.ilabel-1].second; // arc
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

void ConvertFstToLattice(fst::StdVectorFst *net, kaldi::Lattice *lat) {
  for(int32 i = 0; i < net->NumStates(); i++) lat->AddState();
  lat->SetStart(net->Start());
  for(fst::StateIterator<fst::StdVectorFst> siter(*net); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    Weight w_final = net->Final(s);
    if (w_final != Weight::Zero()) { // final state
      kaldi::LatticeWeight new_weight(kaldi::LatticeWeight::One());
      new_weight.SetValue1(w_final.Value());
      lat->SetFinal(s, new_weight);
    }
    for (fst::MutableArcIterator<fst::StdVectorFst> aiter(net, s); !aiter.Done(); aiter.Next()) {
      fst::StdArc arc = aiter.Value();
      kaldi::LatticeWeight new_weight(kaldi::LatticeWeight::One());
      new_weight.SetValue1(arc.weight.Value());
      kaldi::LatticeArc new_arc(arc.ilabel, arc.olabel, new_weight, arc.nextstate);
      lat->AddArc(s, new_arc);
    }
  }
}

void ConvertLatticeToFst(kaldi::Lattice *lat, kaldi::BaseFloat acoustic_scale,
                         fst::StdVectorFst *net) {
  for(int32 i = 0; i < lat->NumStates(); i++) net->AddState();
  net->SetStart(lat->Start());
  for(fst::StateIterator<kaldi::Lattice> siter(*lat);
      !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    kaldi::LatticeWeight w_final = lat->Final(s);
    if (w_final != kaldi::LatticeWeight::Zero()) { // final state
      Weight new_weight( w_final.Value2()/acoustic_scale + w_final.Value1() );
      net->SetFinal(s, new_weight);
    }
    for (fst::MutableArcIterator<kaldi::Lattice> aiter(lat, s);
         !aiter.Done(); aiter.Next()) {
      kaldi::LatticeArc arc = aiter.Value();
      Weight new_weight( arc.weight.Value2()/acoustic_scale + arc.weight.Value1() );
      fst::StdArc new_arc(arc.ilabel, arc.olabel, new_weight, arc.nextstate);
      net->AddArc(s, new_arc);
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

    //BaseFloat acoustic_scale = 0.1;
    //BaseFloat lat_beam = 10.0;
    std::string lattice_wspecifier;
    std::string fst_out_filename;

    const char *usage =
        "Compose decoding graph with given lattices to obtain active-arc lattices.\n"
        "Usage: lattice-arcgraph [options] <tree-in> <topo-file or model-file> "
        "<decoding-graph-fst> <lattice-rspecifier> <arcs-wspecifier>\n"
        " e.g.: lattice-arcgraph tree final.mdl HCLG.fst ark:in.lats ark:out.arcs\n";
      
    ParseOptions po(usage);
    bool topo = false;
    po.Register("topo-file", &topo, "Read topo file instead of model file.");
    po.Register("write-lattices", &lattice_wspecifier, 
      "Write intermediate lattices to archive.");
    po.Register("write-graph", &fst_out_filename,
      "Write intermediate graph to file.");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        tree_filename = po.GetArg(1),
        topo_filename = po.GetArg(2),
        fst_in_filename = po.GetArg(3),
        lats_rspecifier = po.GetArg(4),
        arcs_wspecifier = po.GetArg(5);
    
    // options for lattice determinization
    fst::DeterminizeLatticeOptions lat_opts;
    lat_opts.max_mem = 50000000; // 50 MB
    lat_opts.max_loop = 500000;
    lat_opts.delta = fst::kDelta;

    // load context dependency (tree)
    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);

    // load topograhpy/model and create transition model
    bool binary_in;
    Input ki(topo_filename, &binary_in);
    HmmTopology *topology;
    if (topo) { 
      topology = new HmmTopology;
      topology->Read(ki.Stream(), binary_in);
    } else {
      kaldi::TransitionModel tm;
      tm.Read(ki.Stream(), binary_in);
      topology = new HmmTopology(tm.GetTopo());
    }
    kaldi::TransitionModel trans_model(ctx_dep, *topology);

    // read decoding graph and convert to transition-state acceptor lattice
    fst::StdVectorFst *hclg = fst::ReadFstKaldi(fst_in_filename);
    Lattice graph;
    // convert transition ids (input labels) to transition states
    // and encode graph state in weight or output label
    std::vector<std::pair<int32,int32> > arc_map; // maps state/arc to symbol-id
    ConvertFstToArcLattice(hclg, &graph, &arc_map, &trans_model, true); // keep weights
    fst::ArcSort(&graph, fst::ILabelCompare<kaldi::LatticeArc>());

    //if (fst_out_filename != "") WriteFstKaldi(graph, fst_out_filename);

    //LatticeWriter arcs_writer(arcs_wspecifier);
    TableWriter<fst::VectorFstHolder> arcs_writer(arcs_wspecifier);

    LatticeWriter lat_writer;
    //TableWriter<fst::VectorFstHolder> lat_writer;
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

      // reverse lattice in time
      Lattice lat_reverse;
      fst::Reverse(lat_mapped, &lat_reverse);
      RemoveEpsLocal(&lat_reverse);
      // weight pushing (to make it stochastic) doesn't work with Lattice
      //fst::PushInLog<fst::REWEIGHT_TO_INITIAL>(&lat_reverse, fst::kPushWeights, lat_opts.delta, removetotalweight);
      //fst::ArcSort(&lat_reverse, fst::StdOLabelCompare()); //sufficient if one is sorted
      //if (lattice_wspecifier != "") lat_writer.Write(key, lat_reverse);

      KALDI_LOG << "compose with lattice " << key;
      Lattice lat_composed;
      Compose(lat_reverse, graph, &lat_composed); // composed FST contains HCLG arcs
      if (lattice_wspecifier != "") lat_writer.Write(key, lat_composed);

      CompactLattice clat_determinized;
      if (DeterminizeLattice(lat_composed, &clat_determinized, lat_opts, NULL)) { 
        // now we can forget about the weights
        ScaleLattice(fst::LatticeScale(0.0, 0.0), &clat_determinized);
        Lattice lat_det;
        ConvertLattice(clat_determinized, &lat_det);
        //if (lattice_wspecifier != "") lat_writer.Write(key, lat_det);

        // convert to StdFst, remove word labels
        fst::VectorFst<fst::StdArc> fst_det;
        fst::LatticeToStdMapper<BaseFloat> mapper;
        Map(lat_det, &fst_det, mapper);
        Project(&fst_det, fst::PROJECT_INPUT);
        ArcSort(&fst_det, fst::ILabelCompare<fst::StdArc>()); //to improve speed
        //if (lattice_wspecifier != "") lat_writer.Write(key, fst_det);

        // determinize again on the arc_ids
        fst::VectorFst<fst::StdArc> fst_final;
        bool debug_location = false;
        DeterminizeStar(fst_det, &fst_final, fst::kDelta, &debug_location, -1);
        DecodeGraphSymbols(&fst_final, &arc_map);
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
