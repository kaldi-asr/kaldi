// latbin/nbest-to-prons.cc

// Copyright 2014  Johns Hopkins University (Author: Daniel Povey)

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
#include "lat/lattice-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Reads lattices which must be linear (single path), and must be in\n"
        "CompactLattice form where the transition-ids on the arcs\n"
        "have been aligned with the word boundaries (see lattice-align-words*)\n"
        "and outputs a vaguely ctm-like format where each line is of the form:\n"
        "<utterance-id> <begin-frame> <num-frames> <word> <phone1> <phone2> ... <phoneN>\n"
        "where the words and phones will both be written as integers.  For arcs\n"
        "in the input lattice that don't correspond to words, <word> may be zero; this\n"
        "will typically be the case for the optional silences.\n"
        "\n"
        "Usage: nbest-to-prons [options] <model> <aligned-linear-lattice-rspecifier> <output-wxfilename>\n"
        "e.g.: lattice-1best --acoustic-weight=0.08333 ark:1.lats | \\\n"
        "      lattice-align-words data/lang/phones/word_boundary.int exp/dir/final.mdl ark:- ark:- | \\\n"
        "      nbest-to-prons exp/dir/final.mdl ark:- 1.prons\n"
        "Note: the type of the model doesn't matter as only the transition-model is read.\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        lats_rspecifier = po.GetArg(2),
        wxfilename = po.GetArg(3);


    TransitionModel trans_model;
    ReadKaldiObject(model_rxfilename, &trans_model);
    
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    
    int32 n_done = 0, n_err = 0;

    Output ko(wxfilename, false); // false == non-binary write mode.
    
    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string utt = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();

      std::vector<int32> words, times, lengths;
      std::vector<std::vector<int32> > prons;

      if (!CompactLatticeToWordProns(trans_model, clat, &words, &times, &lengths,
                                     &prons)) {
        n_err++;
        KALDI_WARN << "Format conversion failed for utterance " << utt;
      } else {
        KALDI_ASSERT(words.size() == times.size() &&
                     words.size() == lengths.size() &&
                     words.size() == prons.size());
        for (size_t i = 0; i < words.size(); i++) {
          if (words[i] == 0)  // Don't output anything for <eps> links, which
            continue; // correspond to silence....
          ko.Stream() << utt << ' ' << times[i] << ' ' << lengths[i] << ' '
                      << words[i];
          for (size_t j = 0; j < prons[i].size(); j++)
            ko.Stream() << ' ' << prons[i][j];
          ko.Stream() << std::endl;
        }
        n_done++;
      }
    }
    ko.Close(); // Note: we don't normally call Close() on these things,
    // we just let them go out of scope and it happens automatically.
    // We do it this time in order to avoid wrongly printing out a success message
    // if the stream was going to fail to close
            
    KALDI_LOG << "Printed prons for " << n_done << " linear lattices; "
              << n_err  << " had errors.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
