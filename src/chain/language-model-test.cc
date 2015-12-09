// chain/language-model-test.cc

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

#include "chain/language-model.h"
#include "fstext/fstext-utils.h"

namespace kaldi {
namespace chain {

static void GetTestingData(int32 *vocab_size,
                           std::vector<std::vector<int32> > *data) {
  // read the code of a C++ file as training data.
  bool binary;
  Input input("language-model.cc", &binary);
  KALDI_ASSERT(!binary);
  std::istream &is = input.Stream();
  std::string line;
  *vocab_size = 127;
  int32 line_count = 0;
  for (; getline(is, line); line_count++) {
    std::vector<int32> int_line(line.size());
    for (size_t i = 0; i < line.size(); i++) {
      int32 this_char = line[i];
      if (this_char == 0) {
        this_char = 1;  // should never happen, but just make sure, as 0 is
                        // treated as BOS/EOS in the language modeling code.
      }
      int_line[i] = std::min<int32>(127, this_char);
    }
    data->push_back(int_line);
  }
  KALDI_ASSERT(line_count > 0);
}


void ShowPerplexity(const fst::StdVectorFst &fst,
                    const std::vector<std::vector<int32> > &data) {
  int64 num_phones = 0;
  double tot_loglike = 0;
  for (size_t i = 0; i < data.size(); i++) {
    num_phones += data[i].size();
    fst::StdVectorFst linear_fst;
    MakeLinearAcceptor(data[i], &linear_fst);
    fst::StdVectorFst composed_fst;
    fst::Compose(linear_fst, fst, &composed_fst);
    fst::TropicalWeight weight = fst::ShortestDistance(composed_fst);
    KALDI_ASSERT(weight != fst::TropicalWeight::Zero());
    tot_loglike -= weight.Value();
  }
  double perplexity = exp(-(tot_loglike / num_phones));
  KALDI_LOG << "Perplexity over " << num_phones
            << " phones (of training data) is " << perplexity;
}


void LanguageModelTest() {
  int32 vocab_size;
  std::vector<std::vector<int32> > data;
  GetTestingData(&vocab_size, &data);

  LanguageModelOptions opts;
  opts.no_prune_ngram_order = RandInt(1, 3);
  opts.ngram_order = opts.no_prune_ngram_order + RandInt(0, 3);
  opts.num_extra_lm_states = RandInt(1, 200);
  if (opts.ngram_order < 2)
    opts.ngram_order = 2;
  if (RandInt(1, 2) == 1)
    opts.num_extra_lm_states *= 10;

  LanguageModelEstimator estimator(opts);
  for (size_t i = 0; i < data.size(); i++) {
    std::vector<int32> &sentence = data[i];
    estimator.AddCounts(sentence);
  }

  fst::StdVectorFst fst;
  estimator.Estimate(&fst);
  bool ans = IsStochasticFstInLog(fst);
  KALDI_ASSERT(ans);  // check that it normalizes.
  KALDI_ASSERT(fst.Properties(fst::kAcceptor, true) == fst::kAcceptor);
  KALDI_ASSERT(fst.Properties(fst::kIDeterministic, true) == fst::kIDeterministic);
  KALDI_ASSERT(fst.Properties(fst::kIEpsilons, true) == 0);

  ShowPerplexity(fst, data);
}



}  // namespace chain
}  // namespace kaldi

int main() {
  //  kaldi::SetVerboseLevel(2);
  for (int32 i = 0; i < 30; i++)
    kaldi::chain::LanguageModelTest();
}
