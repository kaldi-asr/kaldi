// latbin/lattice-oracle.cc

// Copyright 2011 Gilles Boulianne
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

using namespace kaldi;

typedef std::vector<fst::StdArc::Label> SymbolVector;

SymbolVector *GetSymbolSet(const fst::StdVectorFst &fst, bool inputSide) {
  SymbolVector *sv = new SymbolVector();
  for (fst::StateIterator<fst::StdVectorFst> siter(fst); !siter.Done(); siter.Next()) {
    fst::StdArc::StateId s = siter.Value();
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, s); !aiter.Done();  aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      sv->push_back(inputSide ? arc.ilabel : arc.olabel);
    }
  }  
  return sv;
}

#ifdef UNUSED
// mapping FST symbols to be ignored to epsilon
void mapIgnoredSymbols(std::unordered_map *ignored_syms, fst::StdArc::Label highest, fst::StdVectorFst *ofst) {
  fst::StdVectorFst *mapping = new fst::StdVectorFst();
  mapping->AddState();
  mapping->SetStart(0);
  mapping->SetFinal(0,fst::StdArc::Weight::One());
  for (fst::StdArc::Label key=0; key <= highest; key++) {
    fst::StdArc::Label mapto;
    mapto = (!ignored_syms || ignored_syms->Find(key) == "") ? key : 0;  // map to 0 if found in table 
    mapping->AddArc(0, fst::StdArc(key,key,0.0,0));
  }
  fst::ArcSort(mapping, fst::StdILabelCompare());
  fst::StdVectorFst origfst(*ofst);
  fst::Compose(origfst,*mapping,ofst);
  fst::RmEpsilon(ofst);
  fst::ArcSort(ofst, fst::StdILabelCompare());
  delete mapping;
}
#endif

// convert from Lattice to standard FST
void convertLatticeToUnweightedAcceptor(const kaldi::Lattice& ilat, 
                                        fst::StdVectorFst *ofst) {
  // first convert from  lattice to normal FST
  //cerr << " convertLatticeToUnweightedAcceptor: ";
  fst::ConvertLattice(ilat, ofst); 
  // remove weights, project to output, sort according to input arg
  fst::Map(ofst, fst::RmWeightMapper<fst::StdArc>()); 
  fst::Project(ofst, fst::PROJECT_OUTPUT);  // for some reason the words are on the output side  
  fst::RmEpsilon(ofst);          // don't tolerate epsilons as they screw up tallying of errors vs arcs
  fst::ArcSort(ofst, fst::StdILabelCompare());
  //cerr << " convertLatticeToUnweightedAcceptor: ofst has "<<ofst->NumArcs(0)<<" arcs at initial state"<<endl;
}

void createEditDistance(fst::StdVectorFst &fst1, fst::StdVectorFst &fst2, fst::StdVectorFst *pfst) {
  fst::StdArc::Weight corrCost(0.0);
  fst::StdArc::Weight subsCost(1.0);
  fst::StdArc::Weight insCost(1.0);
  fst::StdArc::Weight delCost(1.0);

  // create set of output symbols in fst1
  SymbolVector *fst1syms = GetSymbolSet(fst1,false);
  
  // create set of input symbols in fst2
  SymbolVector *fst2syms = GetSymbolSet(fst2,true);

  pfst->AddState();
  pfst->SetStart(0);
  for (SymbolVector::iterator it=fst1syms->begin(); it<fst1syms->end(); it++) {
    pfst->AddArc(0,fst::StdArc(*it,0,delCost,0));    // deletions
  }
  for (SymbolVector::iterator it=fst2syms->begin(); it<fst2syms->end(); it++) {
    pfst->AddArc(0,fst::StdArc(0,*it,insCost,0));    // insertions
  }
  // stupid implementation O(N^2)
  for (SymbolVector::iterator it1=fst1syms->begin(); it1<fst1syms->end(); it1++) {
    for (SymbolVector::iterator it2=fst2syms->begin(); it2<fst2syms->end(); it2++) {
      fst::StdArc::Weight cost( (*it1) == (*it2) ? corrCost : subsCost);
      pfst->AddArc(0,fst::StdArc((*it1),(*it2),cost,0));    // substitutions
    }
  }
  pfst->SetFinal(0,fst::StdArc::Weight::One());
  fst::ArcSort(pfst, fst::StdILabelCompare());
}

void countErrors(fst::StdVectorFst &fst,
                 unsigned int *corr,
                 unsigned int *subs,
                 unsigned int *ins,
                 unsigned int *del,
                 unsigned int *totwords) {
 
   *corr = *subs = *ins = *del = *totwords = 0;

  // go through the first complete path in fst (there should be only one)
  fst::StdArc::StateId src = fst.Start(); 
  while (fst.Final(src)== fst::StdArc::Weight::Zero()) { // while not final
    (*totwords)++;
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst,src); !aiter.Done(); aiter.Next()) {
      fst::StdArc arc = aiter.Value();
      if (arc.ilabel == 0 && arc.olabel == 0) {
        // don't count these so we may compare number of arcs and number of errors
      } else if (arc.ilabel == arc.olabel) {
        (*corr)++;
      } else if (arc.ilabel == 0) {
        (*ins)++;
      } else if (arc.olabel == 0) {
        (*del)++;
      } else {
        (*subs)++;
      }
      src = arc.nextstate;
      continue; // jump to next state
    }
  }
}


bool checkFst(fst::StdVectorFst &fst, string name, string key) {

#ifdef DEBUG
  fst::StdArc::StateId numstates = fst.NumStates();
  cerr << " "<<name<<" has "<<numstates<<" states"<<endl;
  std::stringstream ss; ss <<name<<key<<".fst";
  fst.Write(ss.str());
  return(fst.Start() == fst::kNoStateId); 
#else
  return true;
#endif
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Finds the path having the smallest edit-distance between two lattices.\n"
        "For efficiency put the smallest lattices first (for example reference strings).\n"
        "Usage: lattice-oracle [options] reference-lattice-rspecifier test-lattice-rspecifier transcriptions-wspecifier\n"
        " e.g.: lattice-oracle ark:ref.lats ark:1.lats ark:1.tra\n";
        
    ParseOptions po(usage);
    
    std::string word_syms_filename;

    std::string lats_wspecifier;
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("write-lattices", &lats_wspecifier, "If supplied, write 1-best path as lattices to this wspecifier");
    
    po.Read(argc, argv);
 
    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier1 = po.GetArg(1),
          lats_rspecifier2 = po.GetArg(2),
          transcriptions_wspecifier = po.GetOptArg(3);

    // will read input as  lattices
    SequentialLatticeReader lattice_reader1(lats_rspecifier1);
    RandomAccessLatticeReader lattice_reader2(lats_rspecifier2);
    Int32VectorWriter transcriptions_writer(transcriptions_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_EXIT << "Could not read symbol table from file "
        << word_syms_filename;

    int32 n_done = 0, n_fail = 0;
    unsigned int tot_corr=0, tot_subs=0, tot_ins=0, tot_del=0, tot_words=0;

    for (; !lattice_reader1.Done(); lattice_reader1.Next()) {
      std::string key = lattice_reader1.Key();
      const Lattice lat1 = lattice_reader1.Value();
      cerr << "Lattice "<<key<<" read."<<endl;

      // remove all weights while creating a standard FST
      VectorFst<StdArc> fst1;
      convertLatticeToUnweightedAcceptor(lat1,&fst1);
      checkFst(fst1, "fst1_", key);

      // TODO: map certain symbols (using an FST created with CreateMapFst())
      
      if (!lattice_reader2.HasKey(key)) {
        KALDI_ERR << "No 2nd lattice present for utterance " << key;
      }
      const Lattice lat2 = lattice_reader2.Value(key);
      
      // remove all while creating a normal FST
      VectorFst<StdArc> fst2;
      convertLatticeToUnweightedAcceptor(lat2,&fst2);
      checkFst(fst2, "fst2_", key);
            
      // recreate edit distance fst if necessary
      fst::StdVectorFst editDistanceFst;
      createEditDistance(fst1, fst2, &editDistanceFst);
      
      // compose with edit distance transducer
      VectorFst<StdArc> composedFst;
      fst::Compose(fst1, editDistanceFst, &composedFst);
      checkFst(composedFst, "composed_", key);
      
      // make sure composed FST in output sorted
      fst::ArcSort(&composedFst, fst::StdOLabelCompare());
      
      // compose with previous result
      VectorFst<StdArc> resultFst;
      fst::Compose(composedFst, fst2, &resultFst);
      checkFst(resultFst, "result_", key);
      
      // find out best path
      VectorFst<StdArc> best_path;
      fst::ShortestPath(resultFst, &best_path);
      checkFst(best_path, "best_path_", key);

      if (best_path.Start() == fst::kNoStateId) {
        KALDI_WARN << "Best-path failed for key " << key;
        n_fail++;
      } else {

        // count errors
        unsigned int corr, subs, ins, del, totwords;
        countErrors(best_path, &corr, &subs, &ins, &del, &totwords);
        unsigned int toterrs = subs+ins+del;
        KALDI_LOG << "%WER "<<(100.*toterrs)/totwords<<" [ "<<toterrs<<" / "<<totwords<<", "<<ins<<" ins, "<<del<<" del, "<<subs<<" sub ]";
        tot_corr += corr; tot_subs += subs; tot_ins += ins; tot_del += del; tot_words += totwords;     
        
        std::vector<int32> alignment;
        std::vector<int32> words;
        fst::StdArc::Weight weight;
        GetLinearSymbolSequence(best_path, &alignment, &words, &weight);
        KALDI_LOG << "For utterance " << key << ", best cost " << weight;
        if (transcriptions_wspecifier != "")
          transcriptions_writer.Write(key, words);
        if (word_syms != NULL) {
          std::cerr << key << ' ';
          for (size_t i = 0; i < words.size(); i++) {
            std::string s = word_syms->Find(words[i]);
            if (s == "")
              KALDI_ERR << "Word-id " << words[i] <<" not in symbol table.";
            std::cerr << s << ' ';
          }
          std::cerr << '\n';
        }
      }
      n_done++;
    }
    if (word_syms) delete word_syms;
    unsigned int tot_errs = tot_subs + tot_del + tot_ins;
    KALDI_LOG << "%WER "<<(100.*tot_errs)/tot_words<<" [ "<<tot_errs<<" / "<<tot_words<<", "<<tot_ins<<" ins, "<<tot_del<<" del, "<<tot_subs<<" sub ]";
    KALDI_LOG << "Scored " << n_done << " lattices, "<<n_fail<<" not present in hyp.";
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
