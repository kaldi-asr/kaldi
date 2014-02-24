// latbin/lattice-align-words-lexicon.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
#include "lat/kaldi-lattice.h"
#include "lat/word-align-lattice-lexicon.h"
#include "lat/lattice-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using fst::StdArc;
    using kaldi::int32;

    const char *usage =
        "Convert lattices so that the arcs in the CompactLattice format correspond with\n"
        "words (i.e. aligned with word boundaries).  This is the newest form, that\n"
        "reads in a lexicon in integer format, where each line is (integer id of)\n"
        " word-in word-out phone1 phone2 ... phoneN\n"
        "(note: word-in is word before alignment, word-out is after, e.g. for replacing\n"
        "<eps> with SIL or vice versa)\n"
        "Usage: lattice-align-words-lexicon [options] <lexicon-file> <model> <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-align-words-lexicon  --partial-word-label=4324 --max-expand 10.0 --test true \\\n"
        "   data/lang/phones/align_lexicon.int final.mdl ark:1.lats ark:aligned.lats\n";
    
    ParseOptions po(usage);
    bool output_if_error = true;
    bool output_if_empty = false;
    
    po.Register("output-error-lats", &output_if_error, "Output lattices that aligned "
                "with errors (e.g. due to force-out");
    po.Register("output-if-empty", &output_if_empty, "If true: if algorithm gives "
                "error and produces empty output, pass the input through.");
    
    WordAlignLatticeLexiconOpts opts;
    opts.Register(&po);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        align_lexicon_rxfilename = po.GetArg(1),
        model_rxfilename = po.GetArg(2),
        lats_rspecifier = po.GetArg(3),
        lats_wspecifier = po.GetArg(4);

    std::vector<std::vector<int32> > lexicon;
    {
      bool binary_in;
      Input ki(align_lexicon_rxfilename, &binary_in);
      KALDI_ASSERT(!binary_in && "Not expecting binary file for lexicon");
      if (!ReadLexiconForWordAlign(ki.Stream(), &lexicon)) {
        KALDI_ERR << "Error reading alignment lexicon from "
                  << align_lexicon_rxfilename;
      }
    }

    TransitionModel tmodel;
    ReadKaldiObject(model_rxfilename, &tmodel);
    
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    CompactLatticeWriter clat_writer(lats_wspecifier); 

    WordAlignLatticeLexiconInfo lexicon_info(lexicon);
    { std::vector<std::vector<int32> > temp; lexicon.swap(temp); }
    // No longer needed.
    
    int32 num_done = 0, num_err = 0;
    
    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      const CompactLattice &clat = clat_reader.Value();
      
      CompactLattice aligned_clat;
      
      bool ok = WordAlignLatticeLexicon(clat, tmodel, lexicon_info, opts,
                                        &aligned_clat);
      
      if (!ok) {
        num_err++;
        if (output_if_empty && aligned_clat.NumStates() == 0 &&
            clat.NumStates() != 0) {
          KALDI_WARN << "Algorithm produced no output (due to --max-expand?), "
                     << "so passing input through as output, for key " << key;
          TopSortCompactLatticeIfNeeded(&aligned_clat);
          clat_writer.Write(key, clat);
          continue;
        }
        if (!output_if_error)
          KALDI_WARN << "Lattice for " << key << " did not align correctly";
        else {
          if (aligned_clat.Start() != fst::kNoStateId) {
            KALDI_WARN << "Outputting partial lattice for " << key;
            clat_writer.Write(key, aligned_clat);
          } else {
            KALDI_WARN << "Empty aligned lattice for " << key
                       << ", producing no output.";
          }
        }
      } else {
        if (aligned_clat.Start() == fst::kNoStateId) {
          num_err++;
          KALDI_WARN << "Lattice was empty for key " << key;
        } else {
          num_done++;
          KALDI_VLOG(2) << "Aligned lattice for " << key;
          TopSortCompactLatticeIfNeeded(&aligned_clat);
          clat_writer.Write(key, aligned_clat);
        }
      }
    }
    KALDI_LOG << "Successfully aligned " << num_done << " lattices; "
              << num_err << " had errors.";
    return (num_done > num_err ? 0 : 1); // We changed the error condition slightly here,
    // if there are errors in the word-boundary phones we can get situations
    // where most lattices give an error.
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
