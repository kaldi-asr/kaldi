// latbin/lattice-to-fst.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "hmm/transition-model.h"

namespace kaldi {

void ConvertLatticeToPdfLabels(
    const TransitionModel &tmodel,
    const Lattice &ifst,
    fst::StdVectorFst *ofst) {
  typedef fst::ArcTpl<LatticeWeight> ArcIn;
  typedef fst::StdArc ArcOut;
  typedef ArcIn::StateId StateId;
  ofst->DeleteStates();
  // The states will be numbered exactly the same as the original FST.
  // Add the states to the new FST.
  StateId num_states = ifst.NumStates();
  for (StateId s = 0; s < num_states; s++) {
    StateId news = ofst->AddState();
  }
  ofst->SetStart(ifst.Start());
  for (StateId s = 0; s < num_states; s++) {
    LatticeWeight final_iweight = ifst.Final(s);
    if (final_iweight != LatticeWeight::Zero()) {
      fst::TropicalWeight final_oweight;
      ConvertLatticeWeight(final_iweight, &final_oweight);
      ofst->SetFinal(s, final_oweight);
    }
    for (fst::ArcIterator<Lattice> iter(ifst, s);
         !iter.Done();
         iter.Next()) {
      ArcIn arc = iter.Value();
      KALDI_PARANOID_ASSERT(arc.weight != LatticeWeight::Zero());
      ArcOut oarc;
      ConvertLatticeWeight(arc.weight, &oarc.weight);
      oarc.ilabel = tmodel.TransitionIdToPdf(arc.ilabel) + 1;
      oarc.olabel = arc.olabel;
      oarc.nextstate = arc.nextstate;
      ofst->AddArc(s, oarc);
    }
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
    using std::vector;
    BaseFloat acoustic_scale = 0.0;
    BaseFloat lm_scale = 0.0;
    bool rm_eps = true, read_compact = true, convert_to_pdf_labels = false;
    std::string trans_model;
    bool project_input = false, project_output = true;

    const char *usage =
        "Turn lattices into normal FSTs, retaining only the word labels\n"
        "By default, removes all weights and also epsilons (configure with\n"
        "with --acoustic-scale, --lm-scale and --rm-eps)\n"
        "Usage: lattice-to-fst [options] lattice-rspecifier fsts-wspecifier\n"
        " e.g.: lattice-to-fst  ark:1.lats ark:1.fsts\n";
      
    ParseOptions po(usage);
    po.Register("read-compact", &read_compact, 
                "Read compact lattice. Make this false to convert a "
                "non-compact lattice to an FST. --project-input or --project-output "
                "can be used to get FSA transition-ids or word labels as labels."
                "--convert-to-pdf-labels can be used to convert transition-ids to pdf-id+1");
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale, "Scaling factor for graph/lm costs");
    po.Register("rm-eps", &rm_eps, "Remove epsilons in resulting FSTs (in lazy way; may not remove all)");
    po.Register("convert-to-pdf-labels", &convert_to_pdf_labels,
                "Convert lattice to FST with pdf (1-indexed i.e. pdf_id+1) labels "
                "at the input side; "
                "applicable only when --read-compact=false. "
                "Also supply --project-input to get FSA of pdf_id+1 labels.");
    po.Register("trans-model", &trans_model,
                "Transition model. This is only required if "
                "--convert-to-pdf-labels is true");
    po.Register("project-input", &project_input,
                "Project to input labels (transition-ids); applicable only "
                "when --read-compact=false");
    po.Register("project-output", &project_output,
                "Project to output labels (transition-ids); applicable only "
                "when --read-compact=false");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    vector<vector<double> > scale = fst::LatticeScale(lm_scale, acoustic_scale);
    
    std::string lats_rspecifier = po.GetArg(1),
        fsts_wspecifier = po.GetArg(2);
    
    TransitionModel tmodel;
    if (!trans_model.empty()) {
      ReadKaldiObject(trans_model, &tmodel);
    }

    SequentialCompactLatticeReader compact_lattice_reader;
    SequentialLatticeReader lattice_reader;

    TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);
    
    int32 n_done = 0; // there is no failure mode, barring a crash.
    
    if (read_compact) {
      SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
      for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
        std::string key = compact_lattice_reader.Key();
        CompactLattice clat = compact_lattice_reader.Value();
        compact_lattice_reader.FreeCurrent();
        ScaleLattice(scale, &clat); // typically scales to zero.
        RemoveAlignmentsFromCompactLattice(&clat); // remove the alignments...
        fst::VectorFst<StdArc> fst;
        {
          Lattice lat;
          ConvertLattice(clat, &lat); // convert to non-compact form.. won't introduce
          // extra states because already removed alignments.
          
          ConvertLattice(lat, &fst);

          Project(&fst, fst::PROJECT_OUTPUT); // Because in the standard compact_lattice format,
          // the words are on the output, and we want the word labels.
        }
        if (rm_eps) RemoveEpsLocal(&fst);
        
        fst_writer.Write(key, fst);
        n_done++;
      }
    } else {
      SequentialLatticeReader lattice_reader(lats_rspecifier);
      for (; !lattice_reader.Done(); lattice_reader.Next()) {
        std::string key = lattice_reader.Key();
        Lattice lat = lattice_reader.Value();
        lattice_reader.FreeCurrent();
        ScaleLattice(scale, &lat); // typically scales to zero.
        fst::VectorFst<StdArc> fst;
        if (convert_to_pdf_labels) {
          ConvertLatticeToPdfLabels(tmodel, lat, &fst);
        } else {
          ConvertLattice(lat, &fst);
        }
        if (project_input) 
          Project(&fst, fst::PROJECT_INPUT); 
        else if (project_output)
          Project(&fst, fst::PROJECT_OUTPUT); 
        if (rm_eps) RemoveEpsLocal(&fst);
        
        fst_writer.Write(key, fst);
        n_done++;
      }

    }
    KALDI_LOG << "Done converting " << n_done << " lattices to word-level FSTs";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
