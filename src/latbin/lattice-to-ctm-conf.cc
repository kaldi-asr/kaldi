// latbin/lattice-to-ctm-conf.cc

// Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey)

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

#include "util/common-utils.h"
#include "lat/sausages.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Generate 1-best from lattices and convert into ctm with confidences.\n"
        "If --decode-mbr=true, does Minimum Bayes Risk decoding, else normal\n"
        "Maximum A Posteriori (but works out the confidences based on posteriors\n"
        "in the lattice, using the MBR code).  Note: if you don't need confidences,\n"
        "you can do lattice-1best and pipe to nbest-to-ctm. \n"
        "Note: the ctm this produces will be relative to the utterance-id.\n"
        "Note: the times will only be correct if you do lattice-align-words\n"
        "on the input\n"
        "\n"
        "Usage: lattice-to-ctm-conf [options]  lattice-rspecifier ctm-wxfilename\n"
        " e.g.: lattice-to-ctm-conf --acoustic-scale=0.1 ark:1.lats ark:1.tra "
        "ark:/dev/null ark:1.sau\n";
    
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0, inv_acoustic_scale = 1.0, lm_scale = 1.0;
    bool decode_mbr = true;
    BaseFloat frame_shift = 0.01;

    std::string word_syms_filename;
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for "
                "acoustic likelihoods");
    po.Register("inv-acoustic-scale", &inv_acoustic_scale, "An alternative way "
                "of setting the acoustic scale: you can set its inverse.");
    po.Register("lm-scale", &lm_scale, "Scaling factor for language model "
                "probabilities");
    po.Register("decode-mbr", &decode_mbr, "If true, do Minimum Bayes Risk "
                "decoding (else, Maximum a Posteriori)");
    po.Register("frame-shift", &frame_shift, "Time in seconds between frames.\n");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(acoustic_scale == 1.0 || inv_acoustic_scale == 1.0);
    if (inv_acoustic_scale != 1.0)
      acoustic_scale = 1.0 / inv_acoustic_scale;
    
    std::string lats_rspecifier = po.GetArg(1),
        ctm_wxfilename = po.GetArg(2);
    
    // Read as compact lattice.
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);

    Output ko(ctm_wxfilename, false); // false == non-binary writing mode.
    ko.Stream() << std::fixed;  // Set to "fixed" floating point model, where precision() specifies
    // the #digits after the decimal point.
    ko.Stream().precision(2);

    int32 n_done = 0, n_words = 0;
    BaseFloat tot_bayes_risk = 0.0;
    
    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();
      clat_reader.FreeCurrent();
      fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);

      MinimumBayesRisk mbr(clat, decode_mbr);
      
      const std::vector<BaseFloat> &conf = mbr.GetOneBestConfidences();
      const std::vector<int32> &words = mbr.GetOneBest();
      const std::vector<std::pair<BaseFloat, BaseFloat> > &times =
          mbr.GetOneBestTimes();
      KALDI_ASSERT(conf.size() == words.size() && words.size() == times.size());
      for (size_t i = 0; i < words.size(); i++) {
        KALDI_ASSERT(words[i] != 0); // Should not have epsilons.
        ko.Stream() << key << " 1 " << (frame_shift * times[i].first) << ' '
                    << (frame_shift * (times[i].second-times[i].first)) << ' '
                    << words[i] << ' ' << conf[i] << '\n';
      }
      KALDI_LOG << "For utterance " << key << ", Bayes Risk " << mbr.GetBayesRisk()
                << " and per word, " << mbr.GetBayesRisk()/mbr.GetOneBest().size();
      n_done++;
      n_words += mbr.GetOneBest().size();
      tot_bayes_risk += mbr.GetBayesRisk();
    }

    KALDI_LOG << "Done " << n_done << " lattices.";
    KALDI_LOG << "Overall average Bayes Risk per sentence is "
              << (tot_bayes_risk / n_done) << " and per word, "
              << (tot_bayes_risk / n_words);
    
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
