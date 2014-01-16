// latbin/lattice-boost-ali.cc

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
#include "lat/lattice-functions.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Boost graph likelihoods (decrease graph costs) by b * #frame-phone-errors\n"
        "on each arc in the lattice.  Useful for discriminative training, e.g.\n"
        "boosted MMI.  Modifies input lattices.  This version takes the reference\n"
        "in the form of alignments.  Needs the model (just the transitions) to\n"
        "transform pdf-ids to phones.  Takes the --silence-phones option and these\n"
        "phones appearing in the lattice are always assigned zero error, or with the\n"
        "--max-silence-error option, at most this error-count per frame\n"
        "(--max-silence-error=1 is equivalent to not specifying --silence-phones).\n"
        "\n"
        "Usage: lattice-boost-ali [options] model lats-rspecifier ali-rspecifier lats-wspecifier\n"
        " e.g.: lattice-boost-ali --silence-phones=1:2:3 --b=0.05 1.mdl ark:1.lats ark:1.ali ark:boosted.lats\n";

    kaldi::BaseFloat b = 0.05;
    kaldi::BaseFloat max_silence_error = 0.0;
    std::string silence_phones_str;

    kaldi::ParseOptions po(usage);
    po.Register("b", &b, 
                "Boosting factor (more -> more boosting of errors / larger margin)");
    po.Register("max-silence", &max_silence_error,
                "Maximum error assigned to silence phones [c.f. --silence-phones option]."
                "0.0 -> original BMMI paper, 1.0 -> no special silence treatment.");
    po.Register("silence-phones", &silence_phones_str,
                "Colon-separated list of integer id's of silence phones, e.g. 46:47");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    
    std::vector<int32> silence_phones;
    if (!kaldi::SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones))
      KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    kaldi::SortAndUniq(&silence_phones);
    if (silence_phones.empty())
      KALDI_WARN <<"No silence phones specified, make sure this is what you intended.";
    
    std::string model_rxfilename = po.GetArg(1),
        lats_rspecifier = po.GetArg(2),
        ali_rspecifier = po.GetArg(3),
        lats_wspecifier = po.GetArg(4);

    // Read as regular lattice and write as compact.
    kaldi::SequentialLatticeReader lattice_reader(lats_rspecifier);
    kaldi::RandomAccessInt32VectorReader alignment_reader(ali_rspecifier);
    kaldi::CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    kaldi::TransitionModel trans;
    {
      bool binary_in;
      kaldi::Input ki(model_rxfilename, &binary_in);
      trans.Read(ki.Stream(), binary_in);
    }
    
    int32 n_done = 0, n_err = 0, n_no_ali = 0;
    
    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      kaldi::Lattice lat = lattice_reader.Value();
      lattice_reader.FreeCurrent();

      if (lat.Start() == fst::kNoStateId) {
        KALDI_WARN << "Empty lattice for utterance " << key;
        n_err++;
        continue;
      }
      
      if (b != 0.0) {
        if (!alignment_reader.HasKey(key)) {
          KALDI_WARN << "No alignment for utterance " << key;
          n_no_ali++;
          continue;
        }
        const std::vector<int32> &alignment = alignment_reader.Value(key);
        if (!LatticeBoost(trans, alignment, silence_phones, b,
                          max_silence_error, &lat)) {
          n_err++; // will already have printed warning.
          continue;
        }
      }
      kaldi::CompactLattice clat;
      ConvertLattice(lat, &clat);
      compact_lattice_writer.Write(key, clat);
      n_done++;
    }
    KALDI_LOG << "Done " << n_done << " lattices, missing alignments for "
              << n_no_ali << ", other errors on " << n_err;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
