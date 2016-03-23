// bin/compute-wer-bootci.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University (authors: Jan Trmal, Daniel Povey)
//                2015  Brno Universiry of technology (author: Karel Vesely)
//                2016  Nicolas Serrano

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
#include "util/parse-options.h"
#include "tree/context-dep.h"
#include "util/edit-distance.h"
#include "base/kaldi-math.h"

namespace kaldi {

void GetEditsSingleHyp( const std::string &hyp_rspecifier,
      const std::string &ref_rspecifier,
      const std::string &mode,
      std::vector<std::pair<int32, int32> > & edit_word_per_hyp) {

    // Both text and integers are loaded as vector of strings,
    SequentialTokenVectorReader ref_reader(ref_rspecifier);
    RandomAccessTokenVectorReader hyp_reader(hyp_rspecifier);
    int32 num_words = 0, word_errs = 0, num_ins = 0, num_del = 0, num_sub = 0;

    // Main loop, store WER stats per hyp,
    for (; !ref_reader.Done(); ref_reader.Next()) {
      std::string key = ref_reader.Key();
      const std::vector<std::string> &ref_sent = ref_reader.Value();
      std::vector<std::string> hyp_sent;
      if (!hyp_reader.HasKey(key)) {
        if (mode == "strict")
          KALDI_ERR << "No hypothesis for key " << key << " and strict "
              "mode specifier.";
        if (mode == "present")  // do not score this one.
          continue;
      } else {
        hyp_sent = hyp_reader.Value(key);
      }
      num_words = ref_sent.size();
      word_errs = LevenshteinEditDistance(ref_sent, hyp_sent,
                                            &num_ins, &num_del, &num_sub);
      edit_word_per_hyp.push_back(std::pair<int32, int32>(word_errs, num_words));
    }
}

void GetEditsDualHyp(const std::string &hyp_rspecifier,
      const std::string &hyp_rspecifier2,
      const std::string &ref_rspecifier,
      const std::string &mode,
      std::vector<std::pair<int32, int32> > & edit_word_per_hyp,
      std::vector<std::pair<int32, int32> > & edit_word_per_hyp2) {

    // Both text and integers are loaded as vector of strings,
    SequentialTokenVectorReader ref_reader(ref_rspecifier);
    RandomAccessTokenVectorReader hyp_reader(hyp_rspecifier);
    RandomAccessTokenVectorReader hyp_reader2(hyp_rspecifier2);
    int32 num_words = 0, word_errs = 0,
            num_ins = 0, num_del = 0, num_sub = 0;

    // Main loop, store WER stats per hyp,
    for (; !ref_reader.Done(); ref_reader.Next()) {
      std::string key = ref_reader.Key();
      const std::vector<std::string> &ref_sent = ref_reader.Value();
      std::vector<std::string> hyp_sent, hyp_sent2;
      if (mode == "strict" &&
              (!hyp_reader.HasKey(key) || !hyp_reader2.HasKey(key))) {
          KALDI_ERR << "No hypothesis for key " << key << " in both transcripts "
              "comparison is not possible.";
      } else if (mode == "present" &&
              (!hyp_reader.HasKey(key) || !hyp_reader2.HasKey(key)))
          continue;

      num_words = ref_sent.size();

      //all mode, if a hypothesis is not present, consider as an error
      if(hyp_reader.HasKey(key)){
        hyp_sent = hyp_reader.Value(key);
        word_errs = LevenshteinEditDistance(ref_sent, hyp_sent,
                                            &num_ins, &num_del, &num_sub);
      }
      else
        word_errs = num_words;
      edit_word_per_hyp.push_back(std::pair<int32, int32>(word_errs, num_words));

      if(hyp_reader2.HasKey(key)){
        hyp_sent2 = hyp_reader2.Value(key);
        word_errs = LevenshteinEditDistance(ref_sent, hyp_sent2,
                                            &num_ins, &num_del, &num_sub);
      }
      else
        word_errs = num_words;
      edit_word_per_hyp2.push_back(std::pair<int32, int32>(word_errs, num_words));
    }
}

void GetBootstrapWERInterval(
      const std::vector<std::pair<int32, int32> > & edit_word_per_hyp,
      int32 replications,
      BaseFloat *mean, BaseFloat *interval) {
    BaseFloat wer_accum = 0.0, wer_mult_accum = 0.0;

    for (int32 i = 0; i <= replications; ++i) {
      int32 num_words = 0, word_errs = 0;
      for (int32 j = 0; j <= edit_word_per_hyp.size(); ++j) {
        int32 random_pos = kaldi::RandInt(0, edit_word_per_hyp.size());
        word_errs += edit_word_per_hyp[random_pos].first;
        num_words += edit_word_per_hyp[random_pos].second;
        }

      BaseFloat wer_rep = static_cast<BaseFloat>(word_errs) / num_words;
      wer_accum += wer_rep;
      wer_mult_accum += wer_rep*wer_rep;
    }

    // Compute mean WER and std WER
    *mean = wer_accum / replications;
    *interval = 1.96*sqrt(wer_mult_accum/replications-(*mean)*(*mean));
}

void GetBootstrapWERTwoSystemComparison(
      const std::vector<std::pair<int32, int32> > & edit_word_per_hyp,
      const std::vector<std::pair<int32, int32> > & edit_word_per_hyp2,
      int32 replications, BaseFloat *p_improv) {
    int32 improv_accum = 0.0;

    for (int32 i = 0; i <= replications; ++i) {
      int32 word_errs = 0;
      for (int32 j = 0; j <= edit_word_per_hyp.size(); ++j) {
        int32 random_pos = kaldi::RandInt(0, edit_word_per_hyp.size());
        word_errs += edit_word_per_hyp[random_pos].first -
                        edit_word_per_hyp2[random_pos].first;
        }
      if(word_errs > 0)
        ++improv_accum;
    }
    // Compute mean WER and std WER
    *p_improv = static_cast<BaseFloat>(improv_accum) / replications;
}

} //namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
      "Compute a bootstrapping of WER to extract the 95\% confidence interval.\n"
      "Take a reference and a transcription file, in integer or text format,\n"
      "and outputs overall WER statistics to standard output along with its\n"
      "confidence interval using the bootstrap methos of Bisani and Ney.\n"
      "If a second transcription file corresponding to the same reference is\n"
      "provided, a bootstrap comparison of the two transcription is performed\n"
      "to estimate the probability of improvement.\n"
      "\n"
      "Usage: compute-wer-bootci [options] <ref-rspecifier> <hyp-rspecifier> [<hyp2-rspecifier>]\n"
      "E.g.: compute-wer-bootci --mode=present ark:data/train/text ark:hyp_text\n"
      "or compute-wer-bootci ark:data/train/text ark:hyp_text ark:hyp_text2\n"
      "See also: compute-wer\n";

    ParseOptions po(usage);

    std::string mode = "strict";
    po.Register("mode", &mode,
                "Scoring mode: \"present\"|\"all\"|\"strict\":\n"
                "  \"present\" means score those we have transcriptions for\n"
                "  \"all\" means treat absent transcriptions as empty\n"
                "  \"strict\" means die if all in ref not also in hyp");

    int32 replications = 10000;
    po.Register("replications", &replications,
            "Number of replications to compute the intervals");

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ref_rspecifier = po.GetArg(1);
    std::string hyp_rspecifier = po.GetArg(2);
    std::string hyp2_rspecifier = (po.NumArgs() == 3?po.GetArg(3):"");

    if (mode != "strict" && mode != "present" && mode != "all") {
      KALDI_ERR <<
          "--mode option invalid: expected \"present\"|\"all\"|\"strict\", got "
          << mode;
    }

    //Get editions per each utterance
    std::vector<std::pair<int32, int32> > edit_word_per_hyp, edit_word_per_hyp2;
    if(hyp2_rspecifier.empty())
      GetEditsSingleHyp(hyp_rspecifier, ref_rspecifier, mode, edit_word_per_hyp);
    else
      GetEditsDualHyp(hyp_rspecifier, hyp2_rspecifier, ref_rspecifier, mode,
              edit_word_per_hyp, edit_word_per_hyp2);

    //Extract WER for a number of replications of the same size
    //as the hypothesis extracted
    BaseFloat mean_wer = 0.0, interval = 0.0,
              mean_wer2 = 0.0, interval2 = 0.0,
              p_improv = 0.0;

    GetBootstrapWERInterval(edit_word_per_hyp, replications,
            &mean_wer, &interval);

    if(!hyp2_rspecifier.empty()) {
      GetBootstrapWERInterval(edit_word_per_hyp2, replications,
              &mean_wer2, &interval2);

      GetBootstrapWERTwoSystemComparison(edit_word_per_hyp, edit_word_per_hyp2,
             replications, &p_improv);
    }

    // Print the output,
    std::cout.precision(2);
    std::cerr.precision(2);
    std::cout << "Set1: %WER " << std::fixed << 100*mean_wer <<
              " 95\% Conf Interval [ " << 100*mean_wer-100*interval <<
              ", " << 100*mean_wer+100*interval << " ]" << '\n';

    if(!hyp2_rspecifier.empty()) {
        std::cout << "Set2: %WER " << std::fixed << 100*mean_wer2 <<
            " 95\% Conf Interval [ " << 100*mean_wer2-100*interval2 <<
            ", " << 100*mean_wer2+100*interval2 << " ]" << '\n';

        std::cout << "Probability of Set2 improving Set1: " << std::fixed <<
            100*p_improv << '\n';
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
