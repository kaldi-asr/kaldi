// latbin/lattice-align-words-general.cc

// Copyright 2012  Brno University of Technology (Author: Mirko Hannemann)

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
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
//#include "fst/fstlib.h"


typedef fst::StdArc::StateId StateId;
typedef fst::StdArc::Weight Weight;
typedef fst::StdArc::Label Label;

void ConvertFstToLexLattice(fst::StdVectorFst *net, kaldi::Lattice *lat,
                            bool keep_weights) { // kaldi::TransitionModel *tmodel,
  // converts fst to lattice, removes eventually weights, adds special self-loops

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
      Label olabel = arc.olabel; // word labels 
      kaldi::LatticeArc new_arc(ilabel, olabel, new_weight, arc.nextstate);
      lat->AddArc(s, new_arc);
      arc_id ++;
    }
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using fst::StdArc;
    using kaldi::int32;

    const char *usage =
        "Convert lattices so that the arcs in the CompactLattice format correspond with\n"
        "words (i.e. aligned with word boundaries).  Note: this does not produce quite the\n"
        "same output as lattice-align-words; we only guarantee that the beginnings of words\n"
        "are correctly aligned, and there may be epsilon arcs.  This program may change in\n"
        "future.\n"
        "Usage: lattice-align-words-general [options] <lexicon-fst> <model> <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-align-words-general data/lang/L.fst final.mdl ark:1.lats ark:aligned.lats\n";
    
    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        lexicon_rxfilename = po.GetArg(1),
        model_rxfilename = po.GetArg(2),
        lats_rspecifier = po.GetArg(3),
        lats_wspecifier = po.GetArg(4);

    // read lexicon fst and convert to lattice
    fst::StdVectorFst *lexicon = fst::ReadFstKaldi(lexicon_rxfilename);
    Lattice lex;
    ConvertFstToLexLattice(lexicon, &lex, false); // don't keep weights
    fst::ArcSort(&lex, fst::OLabelCompare<kaldi::LatticeArc>()); // for composition

    TransitionModel tmodel;
    ReadKaldiObject(model_rxfilename, &tmodel);
    
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    //LatticeWriter lat_writer(lats_wspecifier); 
    CompactLatticeWriter clat_writer(lats_wspecifier); 

    int32 num_done = 0, num_err = 0;
    
    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      const CompactLattice &clat = clat_reader.Value();
 
      // create word acceptor w lattice as a copy 
      CompactLattice copy_lat = clat;
      RemoveAlignmentsFromCompactLattice(&copy_lat);
      RemoveWeights(&copy_lat);
      Lattice wlat;
      ConvertLattice(copy_lat, &wlat);
      fst::Project(&wlat, fst::PROJECT_OUTPUT); // project on words.

      // compose lexicon with word acceptor.
      Lattice wlat_composed;
      Compose(lex, wlat, &wlat_composed);
      fst::ArcSort(&wlat_composed, fst::ILabelCompare<kaldi::LatticeArc>()); // for composition

      // convert input lattice to phoneme lattice
      Lattice plat;
      ConvertLattice(clat, &plat);
      ConvertLatticeToPhones(tmodel, &plat); // this function replaces words -> phones

      // compose to get the word labels at beginning of words, otherwise conserving the lattice structure
      Lattice final_composed;
      Compose(plat, wlat_composed, &final_composed);
      CompactLattice final_clat;
      ConvertLattice(final_composed, &final_clat);
      if (final_composed.Start() == fst::kNoStateId) {
        num_err++;
        KALDI_WARN << "Lattice was empty for key " << key;
      } else {
        num_done++;
        KALDI_VLOG(2) << "Aligned lattice for " << key;
        clat_writer.Write(key, final_clat);
      }
    }
    KALDI_LOG << "Successfully aligned " << num_done << " lattices; "
              << num_err << " had errors.";
    return (num_done > num_err ? 0 : 1); // Change the error condition slightly here,
    // if there are errors in the word-boundary phones we can get situations where
    // most lattice give an error.
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
