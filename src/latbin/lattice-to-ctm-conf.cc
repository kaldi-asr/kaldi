// latbin/lattice-to-ctm-conf.cc

// Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey)
//                2015  Guoguo Chen

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
#include "util/kaldi-table.h"
#include "lat/sausages.h"
#include <numeric>

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "This tool turns a lattice into a ctm with confidences, based on the\n"
        "posterior probabilities in the lattice.  The word sequence in the\n"
        "ctm is determined as follows.  Firstly we determine the initial word\n"
        "sequence.  In the 3-argument form, we read it from the\n"
        "<1best-rspecifier> input; otherwise it is the 1-best of the lattice.\n"
        "Then, if --decode-mbr=true, we iteratively refine the hypothesis\n"
        "using Minimum Bayes Risk decoding. (Note that the default value of decode_mbr\n"
        "is true. If you provide <1best-rspecifier> from MAP decoding, the output ctm\n"
        "from MBR decoding may be mismatched with the provided 1best hypothesis (the\n"
        "starting point of optimization). If you don't need confidences,\n"
        "you can do lattice-1best and pipe to nbest-to-ctm. The ctm this\n"
        "program produces will be relative to the utterance-id; a standard\n"
        "ctm relative to the filename can be obtained using\n"
        "utils/convert_ctm.pl.  The times produced by this program will only\n"
        "be meaningful if you do lattice-align-words on the input.  The\n"
        "<1-best-rspecifier> could be the output of utils/int2sym.pl or\n"
        "nbest-to-linear.\n"
        "\n"
        "Usage: lattice-to-ctm-conf [options]  <lattice-rspecifier> \\\n"
        "                                          <ctm-wxfilename>\n"
        "Usage: lattice-to-ctm-conf [options]  <lattice-rspecifier> \\\n"
        "                     [<1best-rspecifier> [<times-rspecifier]] <ctm-wxfilename>\n"
        " e.g.: lattice-to-ctm-conf --acoustic-scale=0.1 ark:1.lats 1.ctm\n"
        "   or: lattice-to-ctm-conf --acoustic-scale=0.1 --decode-mbr=false\\\n"
        "                                      ark:1.lats ark:1.1best 1.ctm\n"
        "See also: lattice-mbr-decode, nbest-to-ctm, lattice-arc-post,\n"
        " steps/get_ctm.sh, steps/get_train_ctm.sh and utils/convert_ctm.pl.\n";

    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0, inv_acoustic_scale = 1.0, lm_scale = 1.0;
    BaseFloat frame_shift = 0.01;
    int32 confidence_digits = 2;

    std::string word_syms_filename;
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for "
                "acoustic likelihoods");
    po.Register("inv-acoustic-scale", &inv_acoustic_scale, "An alternative way "
                "of setting the acoustic scale: you can set its inverse.");
    po.Register("lm-scale", &lm_scale, "Scaling factor for language model "
                "probabilities");
    po.Register("frame-shift", &frame_shift, "Time in seconds between frames.");
    po.Register("confidence-digits", &confidence_digits, "Number of decimal digits for confidences in 'ctm'.");


    MinimumBayesRiskOptions mbr_opts;
    mbr_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2 && po.NumArgs() != 3 && po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(acoustic_scale == 1.0 || inv_acoustic_scale == 1.0);
    if (inv_acoustic_scale != 1.0)
      acoustic_scale = 1.0 / inv_acoustic_scale;

    std::string lats_rspecifier, one_best_rspecifier,
                times_rspecifier, ctm_wxfilename;

    if (po.NumArgs() == 2) {
      lats_rspecifier = po.GetArg(1);
      one_best_rspecifier = "";
      ctm_wxfilename = po.GetArg(2);
    } else if (po.NumArgs() == 3) {
      lats_rspecifier = po.GetArg(1);
      one_best_rspecifier = po.GetArg(2);
      ctm_wxfilename = po.GetArg(3);
    } else if (po.NumArgs() == 4) {
      lats_rspecifier = po.GetArg(1);
      one_best_rspecifier = po.GetArg(2);
      times_rspecifier = po.GetArg(3);
      ctm_wxfilename = po.GetArg(4);
    }


    // Ensure the output ctm file is not a wspecifier
    WspecifierType ctm_wx_type;
    ctm_wx_type  = ClassifyWspecifier(ctm_wxfilename, NULL, NULL, NULL);
    if(ctm_wx_type != kNoWspecifier){
        KALDI_ERR << "The output ctm file should not be a wspecifier. "
          << "Please use things like 1.ctm istead of ark:-";
        exit(1);
    }

    // Read as compact lattice.
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);

    RandomAccessInt32VectorReader one_best_reader(one_best_rspecifier);
    RandomAccessBaseFloatPairVectorReader times_reader(times_rspecifier);

    Output ko(ctm_wxfilename, false); // false == non-binary writing mode.
    ko.Stream() << std::fixed;  // Set to "fixed" floating point model, where precision() specifies
    // the #digits after the decimal point.
    ko.Stream().precision(confidence_digits);

    int32 n_done = 0, n_words = 0;
    BaseFloat tot_bayes_risk = 0.0;

    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();
      clat_reader.FreeCurrent();
      fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);

      MinimumBayesRisk *mbr = NULL;

      if (one_best_rspecifier == "") {
        mbr = new MinimumBayesRisk(clat, mbr_opts);
      } else {
        // check,
        if (!one_best_reader.HasKey(key)) {
          KALDI_WARN << "No 1-best present for utterance " << key;
          continue;
        }
        if (times_rspecifier != "" && !times_reader.HasKey(key)) {
          KALDI_WARN << "No 'times' present for utterance " << key;
          continue;
        }
        // get the 'mbr' decoding object,
        if (times_rspecifier == "") {
          const std::vector<int32> &one_best = one_best_reader.Value(key);
          mbr = new MinimumBayesRisk(clat, one_best, mbr_opts); // no 'times',
        } else {
          // with initial 'times' of the bins,
          const std::vector<int32> &one_best = one_best_reader.Value(key);
          const std::vector<std::pair<BaseFloat,BaseFloat> > &times = times_reader.Value(key);
          mbr = new MinimumBayesRisk(clat, one_best, times, mbr_opts);
        }
      }

      const std::vector<BaseFloat> &conf = mbr->GetOneBestConfidences();
      const std::vector<int32> &words = mbr->GetOneBest();
      const std::vector<std::pair<BaseFloat, BaseFloat> > &times =
          mbr->GetOneBestTimes();
      KALDI_ASSERT(conf.size() == words.size() && words.size() == times.size());
      for (size_t i = 0; i < words.size(); i++) {
        KALDI_ASSERT(words[i] != 0 || mbr_opts.print_silence); // Should not have epsilons.
        ko.Stream() << key << " 1 " << (frame_shift * times[i].first) << ' '
                    << (frame_shift * (times[i].second-times[i].first)) << ' '
                    << words[i] << ' ' << conf[i] << '\n';
      }
      KALDI_LOG << "For utterance " << key << ", Bayes Risk "
                << mbr->GetBayesRisk() << ", avg. confidence per-word "
                << std::accumulate(conf.begin(),conf.end(),0.0) / words.size();
      n_done++;
      n_words += mbr->GetOneBest().size();
      tot_bayes_risk += mbr->GetBayesRisk();
      delete mbr;
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
