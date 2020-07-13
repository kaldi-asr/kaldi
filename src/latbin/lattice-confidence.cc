// latbin/lattice-confidence.cc

// Copyright 2013  Johns Hopkins University (Author: Daniel Povey)

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
#include "lat/confidence.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    using fst::VectorFst;
    using fst::StdArc;
    typedef StdArc::StateId StateId;
    
    const char *usage =
        "Compute sentence-level lattice confidence measures for each lattice.\n"
        "The output is simply the difference between the total costs of the best and\n"
        "second-best paths in the lattice (or a very large value if the lattice\n"
        "had only one path).  Caution: this is not necessarily a very good confidence\n"
        "measure.  You almost certainly want to specify the acoustic scale.\n"
        "If the input is a state-level lattice, you need to specify\n"
        "--read-compact-lattice=false, or the confidences will be very small\n"
        "(and wrong).  You can get word-level confidence info from lattice-mbr-decode.\n"
        "\n"
        "Usage: lattice-confidence <lattice-rspecifier> <confidence-wspecifier>\n"
        "E.g.: lattice-confidence --acoustic-scale=0.08333 ark:- ark,t:-\n";

    ParseOptions po(usage);

    bool read_compact_lattice = true;
    kaldi::BaseFloat acoustic_scale = 1.0, lm_scale = 1.0;
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale,
                "Scaling factor for \"graph costs\" (including LM costs)");
    po.Register("read-compact-lattice", &read_compact_lattice,
                "If true, read CompactLattice format; else, read Lattice format "
                "(necessary for state-level lattices that were written in that "
                "format).");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1);
    std::string confidence_wspecifier = po.GetArg(2);
    
    BaseFloatWriter confidence_writer(confidence_wspecifier);
    
    // Output this instead of infinity; I/O for infinity can be problematic.
    const BaseFloat max_output = 1.0e+10; 
    
    int64 num_done = 0, num_empty = 0,
        num_one_sentence = 0, num_same_sentence = 0;
    double sum_neg_exp = 0.0;
    
    
    if (read_compact_lattice) {
      SequentialCompactLatticeReader clat_reader(lats_rspecifier);
      
      for (; !clat_reader.Done(); clat_reader.Next()) {
        CompactLattice clat = clat_reader.Value();
        std::string key = clat_reader.Key();
        // FreeCurrent() is an optimization that prevents the lattice from being
        // copied unnecessarily (OpenFst does copy-on-write).
        clat_reader.FreeCurrent();
        if (acoustic_scale != 1.0 || lm_scale != 1.0)
          fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);

        int32 num_paths;
        std::vector<int32> best_sentence, second_best_sentence;
        BaseFloat confidence;
        confidence = SentenceLevelConfidence(clat, &num_paths,
                                             &best_sentence,
                                             &second_best_sentence);
        if (num_paths == 0) {
          KALDI_WARN << "Lattice for utterance " << key << " is equivalent to "
                     << "the empty lattice.";
          num_empty++;
          continue;
        } else if (num_paths == 1) {
          num_one_sentence++;
        } else if (num_paths == 2 && best_sentence == second_best_sentence) {
          KALDI_WARN << "Best and second-best sentences were identical: "
                     << "confidence is meaningless.  You should call with "
                     << "--read-compact-lattice=false.";
          num_same_sentence++;
        }
        num_done++;
        confidence = std::min(max_output, confidence); // disallow infinity.
        sum_neg_exp += Exp(-confidence); // diagnostic.
        confidence_writer.Write(key, confidence);
      }
    } else {
      SequentialLatticeReader lat_reader(lats_rspecifier);
      
      for (; !lat_reader.Done(); lat_reader.Next()) {
        Lattice lat = lat_reader.Value();
        std::string key = lat_reader.Key();
        // FreeCurrent() is an optimization that prevents the lattice from being
        // copied unnecessarily (OpenFst does copy-on-write).
        lat_reader.FreeCurrent();
        if (acoustic_scale != 1.0 || lm_scale != 1.0)
          fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &lat);
        int32 num_paths;
        std::vector<int32> best_sentence, second_best_sentence;
        BaseFloat confidence;
        confidence = SentenceLevelConfidence(lat, &num_paths,
                                             &best_sentence,
                                             &second_best_sentence);
        if (num_paths == 0) {
          KALDI_WARN << "Lattice for utterance " << key << " is equivalent to "
                     << "the empty lattice.";
          num_empty++;
          continue;
        } else if (num_paths == 1) {
          num_one_sentence++;
        } else if (num_paths == 2 && best_sentence == second_best_sentence) {
          // This would be an error in some algorithm, in this case.
          KALDI_ERR << "Best and second-best sentences were identical.";
        }
        num_done++;
        confidence = std::min(max_output, confidence); // disallow infinity.
        sum_neg_exp += Exp(-confidence); // diagnostic.
        confidence_writer.Write(key, confidence);
      }
    }
        
    KALDI_LOG << "Done " << num_done << " lattices, of which "
              << num_one_sentence << " contained only one sentence. "
              << num_empty << " were equivalent to the empty lattice.";
    if (num_done != 0)
      KALDI_LOG << "Average confidence (averaged in negative-log space) is "
                << -Log(sum_neg_exp / num_done);
    
    if (num_same_sentence != 0) {
      KALDI_WARN << num_same_sentence << " lattices had the same sentence on "
                 << "different paths (likely an error)";
      return 1;
    }
    return (num_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
