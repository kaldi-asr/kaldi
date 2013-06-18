// bin/compute-wer.cc

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
#include "util/parse-options.h"
#include "tree/context-dep.h"
#include "util/edit-distance.h"

#include "compute-wer.h"


int compute_wer_like_main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Compute WER by comparing different transcriptions\n"
        "Takes two transcription files, in kaldi integer format\n"
        "Usage: compute-wer [options] <ref-rspecifier> <hyp-rspecifier>\n";
    ParseOptions po(usage);

    std::string mode = "strict";
    bool text_input = false;  //  if this is true, we expect symbols as strings,

    po.Register("mode", &mode,
                "Scoring mode: \"present\"|\"all\"|\"strict\":\n"
                "  \"present\" means score those we have transcriptions for\n"
                "  \"all\" means treat absent transcriptions as empty\n"
                "  \"strict\" means die if all in ref not also in hyp");
    po.Register("text", &text_input, "Expect strings, not integers, as input.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string ref_rspecifier = po.GetArg(1);
    std::string hyp_rspecifier = po.GetArg(2);

    if (mode != "strict"
       && mode != "present"
       && mode != "all") {
      KALDI_ERR << "--mode option invalid: expected \"present\"|\"all\"|\"strict\", got "<<mode;
    }



    int32 num_words = 0, word_errs = 0, num_sent = 0, sent_errs = 0,
        num_ins = 0, num_del = 0, num_sub = 0, num_absent_sents = 0;

    if (!text_input) {

      SequentialInt32VectorReader ref_reader(ref_rspecifier);
      RandomAccessInt32VectorReader hyp_reader(hyp_rspecifier);

      for (; !ref_reader.Done(); ref_reader.Next()) {
        std::string key = ref_reader.Key();
        const std::vector<int32> &ref_sent = ref_reader.Value();
        std::vector<int32> hyp_sent;
        if (!hyp_reader.HasKey(key)) {
          if (mode == "strict")
            KALDI_ERR << "No hypothesis for key " << key << " and strict "
                "mode specifier.";
          num_absent_sents++;
          if (mode == "present") // do not score this one.
            continue;
        } else {
          hyp_sent = hyp_reader.Value(key);
        }
        num_words += ref_sent.size();
        int32 ins, del, sub;
        word_errs += LevenshteinEditDistance(ref_sent, hyp_sent, &ins, &del, &sub);
        num_ins += ins; num_del += del; num_sub += sub;

        num_sent++;
        sent_errs += (ref_sent != hyp_sent);
      }
    } else {
      SequentialTokenVectorReader ref_reader(ref_rspecifier);
      RandomAccessTokenVectorReader hyp_reader(hyp_rspecifier);

      for (; !ref_reader.Done(); ref_reader.Next()) {
        std::string key = ref_reader.Key();
        const std::vector<std::string> &ref_sent = ref_reader.Value();
        std::vector<std::string> hyp_sent;
        if (!hyp_reader.HasKey(key)) {
          if (mode == "strict")
            KALDI_ERR << "No hypothesis for key " << key << " and strict "
                "mode specifier.";
          num_absent_sents++;
          if (mode == "present") // do not score this one.
            continue;
        } else {
          hyp_sent = hyp_reader.Value(key);
        }
        num_words += ref_sent.size();
        int32 ins, del, sub;
        word_errs += LevenshteinEditDistance(ref_sent, hyp_sent, &ins, &del, &sub);
        num_ins += ins; num_del += del; num_sub += sub;

        num_sent++;
        sent_errs += (ref_sent != hyp_sent);
      }
    }

    BaseFloat percent_wer = 100.0 * static_cast<BaseFloat>(word_errs)
        / static_cast<BaseFloat>(num_words);
    std::cout.precision(2);
    std::cerr.precision(2);
    std::cout << "%WER "<<std::fixed<<percent_wer<< " [ "<<word_errs<<" / "<<num_words
              <<", "<<num_ins<<" ins, "<<num_del<<" del, "<<num_sub<<" sub ]"
              << (num_absent_sents !=  0 ? " [PARTIAL]" : "") << '\n';
    BaseFloat percent_ser = 100.0 * static_cast<BaseFloat>(sent_errs)
        / static_cast<BaseFloat>(num_sent);
    std::cout << "%SER "<<std::fixed<<percent_ser<< " [ "
              <<sent_errs<<" / "<<num_sent<<" ]\n";
    std::cout << "Scored " << num_sent << " sentences, "
              << num_absent_sents << " not present in hyp.\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


