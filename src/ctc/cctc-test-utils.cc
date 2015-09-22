// ctc/cctc-test-utils.cc

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

#include "ctc/cctc-test-utils.h"

// This test program tests things declared in ctc-supervision.h and cctc-graph.h
// and cctc-training.h, as well as cctc-transition-model.h.

namespace kaldi {
namespace ctc {

void GenerateLanguageModelingData(
    int32 *vocab_size_out,
    std::vector<std::vector<int32> > *data,
    std::vector<std::vector<int32> > *validation_data) {
  bool binary;
  // This test code will probably fail on Windows, as Visual Studio puts
  // binaries in different directories from code, and also filenames are
  // different.  A lot of things fail on Windows, so we're not so concerned
  // about this right now.
  Input input("../ctc/language-model.cc", &binary);
  KALDI_ASSERT(!binary);
  std::istream &is = input.Stream();
  std::string line;
  int32 vocab_size = RandInt(64, 127);
  *vocab_size_out = vocab_size;
  int32 line_count = 0;
  for (; getline(is, line); line_count++) {
    std::vector<int32> int_line(line.size());
    for (size_t i = 0; i < line.size(); i++) {
      int32 this_char = line[i];
      // make sure this_char is in the range [1, vocab_size].
      this_char = (std::abs(this_char - 1) % vocab_size) + 1;
      int_line[i] = this_char;
    }
    if (RandInt(0, 100) == 0)
      int_line.clear();  // Make sure we occasionally generate empty lines.
    if (line_count % 10 != 0)
      data->push_back(int_line);
    else
      validation_data->push_back(int_line);
  }
  KALDI_ASSERT(line_count > 0);
}

// This function, modified from GenRandContextDependency(), generates a random
// context-dependency tree that only has left-context, and ensures that all
// pdf-classes are numbered zero (as required for the CCTC code).
ContextDependency *GenRandContextDependencySpecial(
    const std::vector<int32> &phone_ids) {
  bool ensure_all_covered = true;
  KALDI_ASSERT(IsSortedAndUniq(phone_ids));
  int32 num_stats = 1 + (Rand() % 15) * (Rand() % 15);  // up to 14^2 + 1 separate stats.
  int32 N = 1 + Rand() % 2;  // 1, 2 or 3.  So 0, 1 or 2 phones of left context.
                             //  The transition-model creation code blows up if
                             //  we have more, as it's based on enumerating all
                             //  phone contexts and then merging identical
                             //  history-states.
  int32 P = N - 1;  // Ensure tree left-context only.
  float ctx_dep_prob = 0.7 + 0.3*RandUniform();
  int32 max_phone = *std::max_element(phone_ids.begin(), phone_ids.end());

  std::vector<bool> is_ctx_dep(max_phone + 1);

  std::vector<int32> hmm_lengths(max_phone + 1, -1);

  // I'm guessing the values for i==0 will never be accessed.
  for (int32 i = 1; i <= max_phone; i++) {
    hmm_lengths[i] = 1;
    is_ctx_dep[i] = (RandUniform() < ctx_dep_prob);  // true w.p. ctx_dep_prob.
  }

  // Generate rand stats.
  BuildTreeStatsType stats;
  size_t dim = 3 + Rand() % 20;
  GenRandStats(dim, num_stats, N, P, phone_ids, hmm_lengths,
               is_ctx_dep, ensure_all_covered, &stats);

  // Now build the tree.

  Questions qopts;
  int32 num_quest = Rand() % 10, num_iters = rand () % 5;
  qopts.InitRand(stats, num_quest, num_iters, kAllKeysUnion);  // This was tested in build-tree-utils-test.cc

  float thresh = 100.0 * RandUniform();

  EventMap *tree = NULL;
  std::vector<std::vector<int32> > phone_sets(phone_ids.size());
  for (size_t i = 0; i < phone_ids.size(); i++)
    phone_sets[i].push_back(phone_ids[i]);
  std::vector<bool> share_roots(phone_sets.size(), true),
      do_split(phone_sets.size(), true);

  tree = BuildTree(qopts, phone_sets, hmm_lengths, share_roots,
                   do_split, stats, thresh, 1000, 0.0, P);
  DeleteBuildTreeStats(&stats);
  return new ContextDependency(N, P, tree);
}


void GenerateCctcTransitionModel(CctcTransitionModel *trans_model) {
  int32 order = RandInt(1, 4);
  int32 vocab_size;
  std::vector<std::vector<int32> > data, validation_data;

  GenerateLanguageModelingData(&vocab_size, &data, &validation_data);
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

  KALDI_LOG << "Generating language model with vocab-size = "
            << vocab_size << ", order = " << order << ", cutoffs "
            << opts.state_count_cutoff1 << ","
            << opts.state_count_cutoff2plus << ", perplexity is "
            << ComputePerplexity(lm, validation_data) << "[valid]"
            << " and " << ComputePerplexity(lm, data) << "[train].";

  std::vector<int32> phones;
  for (int32 p = 1; p <= vocab_size; p++)
    phones.push_back(p);
  ContextDependency *dep = GenRandContextDependencySpecial(phones);

  CctcTransitionModelCreator creator(*dep, lm);
  creator.InitCctcTransitionModel(trans_model);
  delete dep;
}


}  // namespace ctc
}  // namespace kaldi

