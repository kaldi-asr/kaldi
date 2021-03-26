// bin/align-text.cc

// Copyright 2014  Guoguo Chen

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
#include "util/parse-options.h"
#include "util/edit-distance.h"
#include <algorithm>

bool IsNotToken(const std::string &token) {
  return ! kaldi::IsToken(token);
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Computes alignment between two sentences with the same key in the\n"
        "two given input text-rspecifiers. The current implementation uses\n"
        "Levenshtein distance as the distance metric.\n"
        "\n"
        "The input text file looks like follows:\n"
        "  key1 a b c\n"
        "  key2 d e\n"
        "\n"
        "The output alignment file looks like follows:\n"
        "  key1 a a ; b <eps> ; c c \n"
        "  key2 d f ; e e \n"
        "where the aligned pairs are separated by \";\"\n"
        "\n"
        "Usage: align-text [options] <text1-rspecifier> <text2-rspecifier> \\\n"
        "                              <alignment-wspecifier>\n"
        " e.g.: align-text ark:text1.txt ark:text2.txt ark,t:alignment.txt\n"
        "See also: compute-wer,\n"
        "Example scoring script: egs/wsj/s5/steps/score_kaldi.sh\n";

    ParseOptions po(usage);

    std::string special_symbol = "<eps>";
    std::string separator = ";";
    po.Register("special-symbol", &special_symbol, "Special symbol to be "
                "aligned with the inserted or deleted words. Your sentences "
                "should not contain this symbol.");
    po.Register("separator", &separator, "Separator for each aligned pair in "
                "the output alignment file.  Note: it should not be necessary "
                "to change this even if your sentences contain ';', because "
                "to parse the output of this program you can just split on "
                "space and then assert that every third token is ';'.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string text1_rspecifier = po.GetArg(1),
        text2_rspecifier = po.GetArg(2),
        align_wspecifier = po.GetArg(3);

    SequentialTokenVectorReader text1_reader(text1_rspecifier);
    RandomAccessTokenVectorReader text2_reader(text2_rspecifier);
    TokenVectorWriter align_writer(align_wspecifier);

    int32 n_done = 0;
    int32 n_fail = 0;
    for (; !text1_reader.Done(); text1_reader.Next()) {
      std::string key = text1_reader.Key();

      if (!text2_reader.HasKey(key)) {
        KALDI_WARN << "Key " << key << " is in " << text1_rspecifier
                   << ", but not in " << text2_rspecifier;
        n_fail++;
        continue;
      }
      const std::vector<std::string> &text1 = text1_reader.Value();
      const std::vector<std::string> &text2 = text2_reader.Value(key);

      if (std::find_if(text1.begin(), text1.end(), IsNotToken) != text1.end()) {
        KALDI_ERR << "In text1, the utterance " << key
                  << " contains unprintable characters. That means there is"
                  << " a problem with the text (such as incorrect encoding).";
      }
      if (std::find_if(text2.begin(), text2.end(), IsNotToken) != text2.end()) {
        KALDI_ERR << "In text2, the utterance " << key
                  << " contains unprintable characters. That means there is"
                  << " a problem with the text (such as incorrect encoding).";
      }

      // Verify that the special symbol is not in the string.
      if (std::find(text1.begin(), text1.end(), special_symbol) != text1.end()){
        KALDI_ERR << "In text1, the utterance " << key
                  << " contains the special symbol '" << special_symbol
                  << "'. This is not allowed.";
      }
      if (std::find(text2.begin(), text2.end(), special_symbol) != text2.end()){
        KALDI_ERR << "In text2, the utterance " << key
                  << " contains the special symbol '" << special_symbol
                  << "'. This is not allowed.";
      }

      std::vector<std::pair<std::string, std::string> > aligned;
      LevenshteinAlignment(text1, text2, special_symbol, &aligned);

      std::vector<std::string> token_vec;
      std::vector<std::pair<std::string, std::string> >::const_iterator iter;
      for (iter = aligned.begin(); iter != aligned.end(); ++iter) {
        token_vec.push_back(iter->first);
        token_vec.push_back(iter->second);
        if (aligned.end() - iter != 1)
          token_vec.push_back(separator);
      }
      align_writer.Write(key, token_vec);

      n_done++;
    }

    KALDI_LOG << "Done " << n_done << " sentences, failed for " << n_fail;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
