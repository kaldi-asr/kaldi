// latbin/lattice-expand-ngram-phone.cc

// Copyright 2015 David Snyder
//                Hossein Hadian  

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
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using kaldi::CompactLatticeArc;

    const char *usage =
      "Expand lattices so that each state has a unique n-label phone history\n"
      "for a specified n. This binary requires that each arc in the\n"
      "input lattices correspond with exactly one phone. The binary\n"
      "lattice-align-phones with the option --remove-epsilon=false can be\n"
      "applied to the lattices to ensure this propery.\n"
      "Usage: lattice-expand-ngram-phone [options] <transition-model> "
      "<lattice-rspecifier> <lattice-wspecifier>\n"
      "e.g.:\n"
      "lattice-expand-ngram-phone --n=4 final.mdl ark:lat ark:expanded_lat\n";

    ParseOptions po(usage);
    int32 n = 4;
    bool test = false;

    std::string word_syms_filename;
    po.Register("n", &n, "The phone n-gram context to expand to.");
    po.Register("test", &test, "If true, verify that the lattices "
                               "have unique left context after expanding.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(n > 0);

    TransitionModel trans_model;
    std::string model_rxfilename = po.GetArg(1),
      lats_rspecifier = po.GetArg(2),
      lats_wspecifier = po.GetOptArg(3);
    ReadKaldiObject(model_rxfilename, &trans_model);

    SequentialCompactLatticeReader lat_reader(lats_rspecifier);
    CompactLatticeWriter lat_writer(lats_wspecifier);

    int32 n_done = 0, n_fail = 0;

    for (; !lat_reader.Done(); lat_reader.Next()) {
      std::string key = lat_reader.Key();
      KALDI_LOG << "Processing lattice for key " << key;
      CompactLattice lat = lat_reader.Value();
      if (test) {
        TopSortCompactLatticeIfNeeded(&lat);
        if (HasUniquePhoneContext(lat, trans_model, n))
          KALDI_LOG << "Before expanding, lattice has unique left context.";
        else
          KALDI_LOG << "Before expanding, lattice does not have "
                    << "unique left context.";
      }
      CompactLattice expanded_lat;
      CompactLatticeExpandByPhone(lat, n, trans_model, &expanded_lat);
      if (expanded_lat.Start() == fst::kNoStateId) {
        KALDI_WARN << "Empty lattice for utterance " << key << std::endl;
       n_fail++;
      } else {
        if (lat.NumStates() == expanded_lat.NumStates()) {
          KALDI_LOG << "Lattice for key " << key
            << " did not need to be expanded for order " << n << ".";
        } else {
          KALDI_LOG << "Lattice expanded from " << lat.NumStates() << " to "
            << expanded_lat.NumStates() << " states for order " << n << ".";
        }
        if (test) {
          TopSortCompactLatticeIfNeeded(&expanded_lat);
          KALDI_ASSERT(HasUniquePhoneContext(expanded_lat, trans_model, n));
          KALDI_LOG << "Lattice has unique left context, after expanding.";
        }
        lat_writer.Write(key, expanded_lat);
        n_done++;
      }
      lat_reader.FreeCurrent();
    }
    KALDI_LOG << "Processed " << n_done << " lattices with " << n_fail
      << " failures.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

