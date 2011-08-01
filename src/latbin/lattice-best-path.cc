// latbin/lattice-best-path.cc

// Copyright 2009-2011  Microsoft Corporation

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Generate 1-best path through lattices; output as transcriptions and alignments\n"
        "Usage: lattice-best-path [options]  lattice-rspecifier [ transcriptions-wspecifier [ alignments-wspecifier] ]\n"
        " e.g.: lattice-best-path --acoustic-scale=0.1 ark:1.lats ark:1.tra ark:1.ali\n";
      
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0;

    std::string word_syms_filename;
    std::string lats_wspecifier;
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("write-lattices", &lats_wspecifier, "If supplied, write 1-best path as lattices to this wspecifier");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        transcriptions_wspecifier = po.GetOptArg(2),
        alignments_wspecifier = po.GetOptArg(3);

    // Read as regular lattice, not compact lattice--
    // this is the form we need it in for efficient best-path.
    SequentialLatticeReader lattice_reader(lats_rspecifier);

    // optional: write 1-best paths as fsts
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier); 
    
    Int32VectorWriter transcriptions_writer(transcriptions_wspecifier);

    Int32VectorWriter alignments_writer(alignments_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_EXIT << "Could not read symbol table from file "
                   << word_syms_filename;


    int32 n_done = 0, n_fail = 0;
    int64 n_frame = 0;
    LatticeWeight tot_weight = LatticeWeight::One();
    
    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      Lattice lat = lattice_reader.Value();
      lattice_reader.FreeCurrent();
      if (acoustic_scale != 1.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);
      Lattice best_path;
      fst::ShortestPath(lat, &best_path);
      if (best_path.Start() == fst::kNoStateId) {
        KALDI_WARN << "Best-path failed for key " << key;
        n_fail++;
      } else {
        std::vector<int32> alignment;
        std::vector<int32> words;
        LatticeWeight weight;
        GetLinearSymbolSequence(best_path, &alignment, &words, &weight);
        KALDI_LOG << "For utterance " << key << ", best cost " <<
            (weight.Value1() + weight.Value2());
        if (transcriptions_wspecifier != "")
          transcriptions_writer.Write(key, words);
        if (alignments_wspecifier != "")
          alignments_writer.Write(key, alignment);
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
        if (lats_wspecifier != "") {
          if (acoustic_scale == 0.0)
            KALDI_ERR << "You can't use zero acoustic scale and write best-path"
                      << " as FSTs (use a very small scale instead).";
          if (acoustic_scale != 1.0)
            fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale),
                              &best_path);
          CompactLattice clat;
          ConvertLattice(best_path, &clat);
          compact_lattice_writer.Write(key, clat);
        }
        n_done++;
        n_frame += alignment.size();
        tot_weight = Times(tot_weight, weight);
      }
    }

    BaseFloat tot_weight_float = tot_weight.Value1() + tot_weight.Value2();
    KALDI_LOG << "Overall score per frame is " << (tot_weight_float/n_frame)
              << " = " << (tot_weight.Value1()/n_frame) << " [graph]"
              << " + " << (tot_weight.Value2()/n_frame) << " [acoustic]"
              << " over " << n_frame << " frames.";
    KALDI_LOG << "Done " << n_done << " lattices, failed for " << n_fail;
    
    if (word_syms) delete word_syms;
    if (n_done != 0) return 0;
    else return 1;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
