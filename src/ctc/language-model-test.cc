// ctc/language-model-test.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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

#include "ctc/language-model.h"

namespace kaldi {
namespace ctc {

void GetTestingData(int32 *vocab_size,
                    std::vector<std::vector<int32> > *data,
                    std::vector<std::vector<int32> > *validation_data) {
  // read the code of a C++ file as training data.
  bool binary;
  Input input("language-model.cc", &binary);
  KALDI_ASSERT(!binary);
  std::istream &is = input.Stream();
  std::string line;
  *vocab_size = 255;
  int32 line_count = 0;
  for (; getline(is, line); line_count++) {
    std::vector<int32> int_line(line.size());
    for (size_t i = 0; i < line.size(); i++) {
      int32 this_char = line[i];
      if (this_char == 0) {
        this_char = 1;  // should never happen, but just make sure, as 0 is
                        // treated as BOS/EOS in the language modeling code.
      }
      int_line[i] = this_char;
    }
    if (line_count % 10 != 0)
      data->push_back(int_line);
    else
      validation_data->push_back(int_line);
  }
  KALDI_ASSERT(line_count > 0);
}


void TestLmHistoryStateMap(const LanguageModel &lm) {
  LmHistoryStateMap map;
  map.Init(lm);
  KALDI_LOG << "Number of history states is " << map.NumLmHistoryStates();

  int32 vocab_size = lm.VocabSize();
  int32 num_test = 500;
  for (int32 i = 0; i < num_test; i++) {
    int32 history_length = RandInt(0, lm.NgramOrder() - 1);
    std::vector<int32> history(history_length);
    // get a random history.
    for (int32 i = 0; i < history_length; i++)
      history[i] = RandInt(0, vocab_size);
    int32 history_state = map.GetLmHistoryState(history);

    std::vector<int32> ngram(history);
    int32 random_word = RandInt(0, vocab_size);
    ngram.push_back(random_word);
    KALDI_ASSERT(map.GetProb(lm, history_state, random_word) ==
                 lm.GetProb(ngram));
  }
}

void TestNormalization(const LanguageModel &lm) {
  int32 vocab_size = lm.VocabSize();
  int32 num_test = 500;
  for (int32 i = 0; i < num_test; i++) {
    int32 history_length = RandInt(0, lm.NgramOrder() - 1);
    std::vector<int32> history(history_length);
    // get a random history.
    for (int32 i = 0; i < history_length; i++)
      history[i] = RandInt(0, vocab_size);
    double prob_sum = 0.0;
    std::vector<int32> vec(history);
    vec.push_back(0);
    for (int32 word = 0; word <= vocab_size; word++) {
      vec[history_length] = word;
      prob_sum += lm.GetProb(vec);
    }
    KALDI_ASSERT(ApproxEqual(prob_sum, 1.0));
  }
}

void LanguageModelTest() {
  int32 order = RandInt(1, 4);
  int32 vocab_size;
  std::vector<std::vector<int32> > data, validation_data;

  GetTestingData(&vocab_size, &data, &validation_data);
  
  LanguageModelOptions opts;
  opts.ngram_order = order;
  if (RandInt(0,3) == 0)
    opts.state_count_cutoff1 = 100.0;
  if (RandInt(0,3) == 0) {
    opts.state_count_cutoff1 = 10.0;
    opts.state_count_cutoff2plus = 10.0;
  }
  if (RandInt(0,5) == 0) {
    opts.state_count_cutoff1 = 0.0;
    opts.state_count_cutoff2plus = 0.0;
  }
  
  
  LanguageModelEstimator estimator(opts, vocab_size);
  for (size_t i = 0; i < data.size(); i++) {
    std::vector<int32> &sentence = data[i];
    estimator.AddCounts(sentence);
  }
  estimator.Discount();
  LanguageModel lm;
  estimator.Output(&lm);


  KALDI_LOG << "For order " << order << ", cutoffs "
            << opts.state_count_cutoff1 << ","
            << opts.state_count_cutoff2plus << ", perplexity is "
            << ComputePerplexity(lm, validation_data) << "[valid]"
            << " and " << ComputePerplexity(lm, data) << "[train].";
  TestNormalization(lm);
  TestLmHistoryStateMap(lm);
}



}  // namespace ctc
}  // namespace kaldi

int main() {
  for (int32 i = 0; i < 30; i++)
    kaldi::ctc::LanguageModelTest();
}
