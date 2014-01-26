// latbin/lattice-oracle.cc

// Copyright 2011 Gilles Boulianne
//           2013 Johns Hopkins University (author: Daniel Povey)
//
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

namespace kaldi {

using std::vector;
using std::set;

typedef unordered_set<fst::StdArc::Label> LabelSet; 

void ReadSymbolList(const std::string &rxfilename,
                    fst::SymbolTable *word_syms,
                    LabelSet *lset) {
  Input ki(rxfilename);
  std::string line;
  KALDI_ASSERT(lset != NULL);
  lset->clear();
  while (getline(ki.Stream(), line)) {
    std::string sym;
    std::istringstream ss(line);
    ss >> sym >> std::ws;
    if (ss.fail() || !ss.eof()) {
      KALDI_ERR << "Bad line in symbol list: "<< line
                << ", file is: " << PrintableRxfilename(rxfilename);
    }
    fst::StdArc::Label lab = word_syms->Find(sym.c_str());
    if (lab == fst::SymbolTable::kNoSymbol) {
      KALDI_ERR << "Can't find symbol in symbol table: "
                << line << ", file is: "
                << PrintableRxfilename(rxfilename);
    }
    lset->insert(lab);
  }
}

void MapWildCards(const LabelSet &wildcards, fst::StdVectorFst *ofst) {
  // map all wildcards symbols to epsilons
  for (fst::StateIterator<fst::StdVectorFst> siter(*ofst); !siter.Done(); siter.Next()) {
    fst::StdArc::StateId s = siter.Value();
    for (fst::MutableArcIterator<fst::StdVectorFst> aiter(ofst, s); !aiter.Done();  aiter.Next()) {
      fst::StdArc arc(aiter.Value());
      LabelSet::const_iterator it = wildcards.find(arc.ilabel);
      if (it != wildcards.end()) {
        KALDI_VLOG(4) << "MapWildCards: mapping symbol " << arc.ilabel
                      << " to epsilon" << endl;
        arc.ilabel = 0;
      }
      it = wildcards.find(arc.olabel);
      if (it != wildcards.end()) {arc.olabel = 0;}
      aiter.SetValue(arc);
    }
  }    
}

// convert from Lattice to standard FST
// also maps wildcard symbols to epsilons
// then removes epsilons
void ConvertLatticeToUnweightedAcceptor(const kaldi::Lattice &ilat,
                                        const LabelSet &wildcards,
                                        fst::StdVectorFst *ofst) {
  // first convert from  lattice to normal FST
  fst::ConvertLattice(ilat, ofst); 
  // remove weights, project to output, sort according to input arg
  fst::Map(ofst, fst::RmWeightMapper<fst::StdArc>()); 
  fst::Project(ofst, fst::PROJECT_OUTPUT);  // The words are on the output side  
  MapWildCards(wildcards, ofst);
  fst::RmEpsilon(ofst);   // Don't tolerate epsilons as they make it hard to tally errors
  fst::ArcSort(ofst, fst::StdILabelCompare());
}

void CreateEditDistance(const fst::StdVectorFst &fst1,
                        const fst::StdVectorFst &fst2,
                        fst::StdVectorFst *pfst) {
  using namespace fst;
  typedef StdArc StdArc;
  typedef StdArc::Weight Weight;
  typedef StdArc::Label Label;
  Weight correct_cost(0.0);
  Weight substitution_cost(1.0);
  Weight insertion_cost(1.0);
  Weight deletion_cost(1.0);

  // create set of output symbols in fst1
  std::vector<Label> fst1syms, fst2syms;
  GetOutputSymbols(fst1, false /*no epsilons*/, &fst1syms);
  GetInputSymbols(fst2, false /*no epsilons*/, &fst2syms);

  pfst->AddState();
  pfst->SetStart(0);
  for (size_t i = 0; i < fst1syms.size(); i++) 
    pfst->AddArc(0, StdArc(fst1syms[i], 0, deletion_cost, 0)); // deletions
  
  for (size_t i = 0; i < fst2syms.size(); i++)
    pfst->AddArc(0, StdArc(0, fst2syms[i], insertion_cost, 0));  // insertions
 
  // stupid implementation O(N^2)
  for (size_t i = 0; i < fst1syms.size(); i++) {
    Label label1 = fst1syms[i];
    for (size_t j = 0; j < fst2syms.size(); j++) {
      Label label2 = fst2syms[j];
      Weight cost( label1 == label2 ? correct_cost : substitution_cost);
      pfst->AddArc(0, StdArc(label1, label2, cost, 0)); // substitutions
    }
  }
  pfst->SetFinal(0, Weight::One());
  ArcSort(pfst, StdOLabelCompare());
}

void CountErrors(fst::StdVectorFst &fst,
                 int32 *correct,
                 int32 *substitutions,
                 int32 *insertions,
                 int32 *deletions,
                 int32 *num_words) {
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Weight Weight;
   *correct = *substitutions = *insertions = *deletions = *num_words = 0;

  // go through the first complete path in fst (there should be only one)
  StateId src = fst.Start(); 
  while (fst.Final(src)== Weight::Zero()) { // while not final
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, src); !aiter.Done(); aiter.Next()) {
      fst::StdArc arc = aiter.Value();
      if (arc.ilabel == arc.olabel && arc.ilabel != 0) {
        (*correct)++;
        (*num_words)++;
      } else if (arc.ilabel == 0 && arc.olabel != 0) {
        (*deletions)++;
        (*num_words)++;
      } else if (arc.ilabel != 0 && arc.olabel == 0) {
        (*insertions)++; 
      } else if (arc.ilabel != 0 && arc.olabel != 0) {
        (*substitutions)++;
        (*num_words)++;
      } else {
        KALDI_ASSERT(arc.ilabel == 0 && arc.olabel == 0);
      }
      src = arc.nextstate;
      continue; // jump to next state
    }
  }
}


bool CheckFst(fst::StdVectorFst &fst, string name, string key) {

#ifdef DEBUG
  StateId numstates = fst.NumStates();
  cerr << " "<<name<<" has "<<numstates<<" states"<<endl;
  std::stringstream ss; ss <<name<<key<<".fst";
  fst.Write(ss.str());
  return(fst.Start() == fst::kNoStateId); 
#else
  return true;
#endif
}


// Guoguo Chen added the implementation for option "write-lattices". This
// function does a depth first search on the lattice and remove the arcs that
// don't correctespond to the oracle path. By "remove" I actually point the next
// state of the arc to some state that is not in the lattice and then use the
// openfst connect function. This makes things much easier. 
bool GetOracleLattice(Lattice *oracle_lat, 
                      vector<int32> oracle_words, 
                      LatticeArc::StateId bad_state,
                      LatticeArc::StateId current_state, 
                      int32 current_word) {
  if (current_word == oracle_words.size()) {
    if (oracle_lat->Final(current_state) != LatticeArc::Weight::Zero())
      return true;
  } else {
    if (oracle_lat->Final(current_state) != LatticeArc::Weight::Zero())
      return false;
  }

  bool status = false;
  for (fst::MutableArcIterator<Lattice> aiter(oracle_lat, current_state);
       !aiter.Done();
       aiter.Next()) {
    LatticeArc arc(aiter.Value());
    LatticeArc::StateId nextstate = arc.nextstate;
    if (arc.olabel == 0)
      status = GetOracleLattice(oracle_lat, oracle_words, bad_state, nextstate, current_word) || status;
    else if (current_word < oracle_words.size() && arc.olabel == oracle_words[current_word])
      status = GetOracleLattice(oracle_lat, oracle_words, bad_state, nextstate, ++current_word) || status;
    else {
      arc.nextstate = bad_state;
      aiter.SetValue(arc);
    }
  }

  if (current_state == oracle_lat->Start())
    fst::Connect(oracle_lat);

  return status;
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
    typedef fst::StdArc::Weight Weight;
    typedef fst::StdArc::StateId StateId;

    const char *usage =
        "Finds the path having the smallest edit-distance between two lattices.\n"
        "For efficiency put the smallest lattices first (for example reference strings).\n"
        "Usage: lattice-oracle [options] test-lattice-rspecifier reference-rspecifier "
        "transcriptions-wspecifier [edit-distance-wspecifier]\n"
        " e.g.: lattice-oracle ark:lat.1 'ark:sym2int.pl -f 2- data/lang/words.txt <data/test/text' ark,t:-\n";
        
    ParseOptions po(usage);
    
    std::string word_syms_filename;
    std::string wild_syms_rxfilename;
    std::string wildcard_symbols;
    std::string lats_wspecifier;
    
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("wildcard-symbols-list", &wild_syms_rxfilename, "Filename (generally, "
                "rxfilename) for file containing text-form list of symbols that "
                "don't count as errors; this option requires --word-symbol-table."
                "  Deprecated; use --wildcard-symbols option.");
    po.Register("wildcard-symbols", &wildcard_symbols,
                "Colon-separated list of integer ids of symbols that "
                "don't count as errors.  Preferred alternative to deprecated "
                "option --wildcard-symbols-list.");
    po.Register("write-lattices", &lats_wspecifier, "If supplied, write 1-best "
                "path as lattices to this wspecifier");
    
    po.Read(argc, argv);
 
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        reference_rspecifier = po.GetArg(2),
        transcriptions_wspecifier = po.GetArg(3);

    // will read input as  lattices
    SequentialLatticeReader lattice_reader(lats_rspecifier);
    RandomAccessInt32VectorReader reference_reader(reference_rspecifier);

    Int32VectorWriter transcriptions_writer(transcriptions_wspecifier);
    
    // Guoguo Chen added the implementation for option "write-lattices".
    CompactLatticeWriter lats_writer(lats_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_filename;

    LabelSet wildcards;
    if (wild_syms_rxfilename != "") {
      KALDI_WARN << "--wildcard-symbols-list option deprecated.";
      KALDI_ASSERT(wildcard_symbols.empty() && "Do not use both "
                   "--wildcard-symbols and --wildcard-symbols-list options.");
      KALDI_ASSERT(word_syms != NULL && "--wildcard-symbols-list option "
                   "requires --word-symbol-table option");
      ReadSymbolList(wild_syms_rxfilename, word_syms, &wildcards);
    } else {
      std::vector<fst::StdArc::Label> wildcard_symbols_vec;
      if (!SplitStringToIntegers(wildcard_symbols, ":", false,
                                 &wildcard_symbols_vec)) {
        KALDI_ERR << "Expected colon-separated list of integers for "
                  << "--wildcard-symbols option, got: " << wildcard_symbols;
      }
      for (size_t i = 0; i < wildcard_symbols_vec.size(); i++)
        wildcards.insert(wildcard_symbols_vec[i]);
    }  
    
    int32 n_done = 0, n_fail = 0;
    int32 tot_correct=0, tot_substitutions=0, tot_insertions=0, tot_deletions=0,
        tot_words=0;

    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      const Lattice &lat = lattice_reader.Value();
      cerr << "Lattice "<<key<<" read."<<endl;

      // remove all weights while creating a standard FST
      VectorFst<StdArc> lattice_fst;
      ConvertLatticeToUnweightedAcceptor(lat, wildcards, &lattice_fst);
      CheckFst(lattice_fst, "lattice_fst_", key);
      
      // TODO: map certain symbols (using an FST created with CreateMapFst())
      
      if (!reference_reader.HasKey(key)) {
        KALDI_WARN << "No reference present for utterance " << key;
        n_fail++;
        continue;
      }
      const std::vector<int32> &reference = reference_reader.Value(key);
      VectorFst<StdArc> reference_fst;
      MakeLinearAcceptor(reference, &reference_fst);
      MapWildCards(wildcards, &reference_fst); // Remove any wildcards in reference.
      
      CheckFst(reference_fst, "reference_fst_", key);
            
      // recreate edit distance fst if necessary
      fst::StdVectorFst edit_distance_fst;
      CreateEditDistance(lattice_fst, reference_fst, &edit_distance_fst);
      
      // compose with edit distance transducer
      VectorFst<StdArc> edit_ref_fst;
      fst::Compose(edit_distance_fst, reference_fst, &edit_ref_fst);
      CheckFst(edit_ref_fst, "composed_", key);
      
      // make sure composed FST is input sorted
      fst::ArcSort(&edit_ref_fst, fst::StdILabelCompare());
      
      // compose with previous result
      VectorFst<StdArc> result_fst;
      fst::Compose(lattice_fst, edit_ref_fst, &result_fst);
      CheckFst(result_fst, "result_", key);
      
      // find out best path
      VectorFst<StdArc> best_path;
      fst::ShortestPath(result_fst, &best_path);
      CheckFst(best_path, "best_path_", key);

      if (best_path.Start() == fst::kNoStateId) {
        KALDI_WARN << "Best-path failed for key " << key;
        n_fail++;
      } else {
        // count errors
        int32 correct, substitutions, insertions, deletions, num_words;
        CountErrors(best_path, &correct, &substitutions, &insertions, &deletions, &num_words);
        int32 toterrs = substitutions + insertions + deletions;
        KALDI_LOG << "%WER " << (100.*toterrs) / num_words << " [ " << toterrs
                  << " / " << num_words << ", " << insertions << " insertions, " << deletions
                  << " deletions, " << substitutions << " sub ]";
        tot_correct += correct;
        tot_substitutions += substitutions;
        tot_insertions += insertions;
        tot_deletions += deletions;
        tot_words += num_words;     
        
        std::vector<int32> oracle_words;
        std::vector<int32> reference_words;
        Weight weight;
        GetLinearSymbolSequence(best_path, &oracle_words, &reference_words, &weight);
        KALDI_LOG << "For utterance " << key << ", best cost " << weight;
        if (transcriptions_wspecifier != "")
          transcriptions_writer.Write(key, oracle_words);
        if (word_syms != NULL) {
          std::cerr << key << " (oracle) ";
          for (size_t i = 0; i < oracle_words.size(); i++) {
            std::string s = word_syms->Find(oracle_words[i]);
            if (s == "")
              KALDI_ERR << "Word-id " << oracle_words[i] <<" not in symbol table.";
            std::cerr << s << ' ';
          }
          std::cerr << '\n' << key << " (reference) ";
          for (size_t i = 0; i < reference_words.size(); i++) {
            std::string s = word_syms->Find(reference_words[i]);
            if (s == "")
              KALDI_ERR << "Word-id " << reference_words[i]
                        << " not in symbol table.";
            std::cerr << s << ' ';
          }
          std::cerr << '\n';
        }

        // Guoguo Chen added the implementation for option "write-lattices".
        // Currently it's just a naive implementation: traversal the original
        // lattice and get the path corresponding to the oracle word sequence.
        // Note that this new lattice has the alignment information.
        if (lats_wspecifier != "") {
          Lattice oracle_lat = lat;
          LatticeArc::StateId bad_state = oracle_lat.AddState();
          if (!GetOracleLattice(&oracle_lat, oracle_words,
                                bad_state, oracle_lat.Start(), 0)) 
            KALDI_WARN << "Failed to find the oracle path in the original "
                       << "lattice: " << key;
          CompactLattice oracle_clat;
          ConvertLattice(oracle_lat, &oracle_clat);
          lats_writer.Write(key, oracle_clat);
        }
      }
      n_done++;
    }
    if (word_syms) delete word_syms;
    int32 tot_errs = tot_substitutions + tot_deletions + tot_insertions;
    // Warning: the script egs/s5/*/steps/oracle_wer.sh parses the next line.
    KALDI_LOG << "Overall %WER " << (100.*tot_errs)/tot_words << " [ "
              << tot_errs << " / " << tot_words << ", " << tot_insertions
              << " insertions, " << tot_deletions << " deletions, "
              << tot_substitutions << " substitutions ]";
    KALDI_LOG << "Scored " << n_done << " lattices, " << n_fail
              << " not present in ref.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
