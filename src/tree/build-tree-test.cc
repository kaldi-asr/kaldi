// tree/build-tree-test.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "util/stl-utils.h"
#include "tree/build-tree.h"

namespace kaldi {

void TestGenRandStats() {
  for (int32 p = 0; p < 2; p++) {
    int32 dim = 1 + rand() % 40;
    int32 num_phones = 1 + rand() % 40;
    int32 num_stats = 1 +  (rand() % 20);
    int32 N = 2 + rand() % 2;  // 2 or 3.
    int32 P = rand() % N;
    float ctx_dep_prob = 0.5 + 0.5*RandUniform();
    std::vector<int32> phone_ids(num_phones);
    for (size_t i = 0;i < (size_t)num_phones;i++)
      phone_ids[i] = (i == 0 ? (rand() % 2) : phone_ids[i-1] + 1 + (rand()%2));
    int32 max_phone = *std::max_element(phone_ids.begin(), phone_ids.end());
    std::vector<int32> hmm_lengths(max_phone+1);
    std::vector<bool> is_ctx_dep(max_phone+1);

    for (int32 i = 0; i <= max_phone; i++) {
      hmm_lengths[i] = 1 + rand() % 3;
      is_ctx_dep[i] = (RandUniform() < ctx_dep_prob);  // true w.p. ctx_dep_prob.
    }
    for (size_t i = 0;i < (size_t) num_phones;i++) {
      KALDI_VLOG(2) <<  "For idx = "<< i << ", (phone_id, hmm_length, is_ctx_dep) == " << (phone_ids[i]) << " " << (hmm_lengths[phone_ids[i]]) << " " << (is_ctx_dep[phone_ids[i]]);
    }
    BuildTreeStatsType stats;
    // put false for all_covered argument.
    // if it doesn't really ensure that all are covered with true, this will induce
    // failure in the test of context-fst.
    GenRandStats(dim, num_stats, N, P, phone_ids, hmm_lengths, is_ctx_dep, false, &stats);
    std::cout << "Writing random stats.";
    std::cout <<"dim = " << dim << '\n';
    std::cout <<"num_phones = " << num_phones << '\n';
    std::cout <<"num_stats = " << num_stats << '\n';
    std::cout <<"N = "<< N << '\n';
    std::cout <<"P = "<< P << '\n';
    std::cout << "is-ctx-dep = ";
    for (size_t i = 0;i < is_ctx_dep.size();i++)
      WriteBasicType(std::cout, false, static_cast<bool>(is_ctx_dep[i]));
    std::cout << "hmm_lengths = "; WriteIntegerVector(std::cout, false, hmm_lengths);
    std::cout << "phone_ids = "; WriteIntegerVector(std::cout, false, phone_ids);
    std::cout << "Stats are: \n";
    WriteBuildTreeStats(std::cout, false, stats);


    // Now check the properties of the stats.
    for (size_t i = 0;i < stats.size();i++) {
      EventValueType central_phone;
      bool b = EventMap::Lookup(stats[i].first, P, &central_phone);
      KALDI_ASSERT(b);
      EventValueType position;
      b = EventMap::Lookup(stats[i].first, kPdfClass, &position);
      KALDI_ASSERT(b);
      KALDI_ASSERT(position>=0 && position < hmm_lengths[central_phone]);

      for (EventKeyType j = 0; j < N; j++) {
        if (j != P) {  // non-"central" phone.
          EventValueType ctx_phone;
          b = EventMap::Lookup(stats[i].first, j, &ctx_phone);
          KALDI_ASSERT(is_ctx_dep[central_phone] == b);
        }
      }
    }
    DeleteBuildTreeStats(&stats);
  }
}


void TestBuildTree() {
  for (int32 p = 0; p < 3; p++) {
    // First decide phone-ids, hmm lengths, is-ctx-dep...

    int32 dim = 1 + rand() % 40;
    int32 num_phones = 1 + rand() % 8;
    int32 num_stats = 1 + (rand() % 15) * (rand() % 15);  // up to 14^2 + 1 separate stats.
    int32 N = 2 + rand() % 2;  // 2 or 3.
    int32 P = rand() % N;
    float ctx_dep_prob = 0.5 + 0.5*RandUniform();

    std::vector<int32> phone_ids(num_phones);
    for (size_t i = 0;i < (size_t)num_phones;i++)
      phone_ids[i] = (i == 0 ? (rand() % 2) : phone_ids[i-1] + 1 + (rand()%2));
    int32 max_phone = *std::max_element(phone_ids.begin(), phone_ids.end());
    std::vector<int32> hmm_lengths(max_phone+1);
    std::vector<bool> is_ctx_dep(max_phone+1);

    for (int32 i = 0; i <= max_phone; i++) {
      hmm_lengths[i] = 1 + rand() % 3;
      is_ctx_dep[i] = (RandUniform() < ctx_dep_prob);  // true w.p. ctx_dep_prob.
    }
    for (size_t i = 0;i < (size_t) num_phones;i++) {
      KALDI_VLOG(2) <<  "For idx = "<< i << ", (phone_id, hmm_length, is_ctx_dep) == " << (phone_ids[i]) << " " << (hmm_lengths[phone_ids[i]]) << " " << (is_ctx_dep[phone_ids[i]]);
    }
    // Generate rand stats.  These were tested in TestGenRandStats() above.
    BuildTreeStatsType stats;
    bool ensure_all_covered = false;
    GenRandStats(dim, num_stats, N, P, phone_ids, hmm_lengths, is_ctx_dep, ensure_all_covered, &stats);

    {  // print out the stats.
      std::cout << "Writing random stats.";
      std::cout << "dim = " << dim << '\n';
      std::cout << "num_phones = " << num_phones << '\n';
      std::cout << "num_stats = " << num_stats << '\n';
      std::cout << "N = "<< N << '\n';
      std::cout << "P = "<< P << '\n';
      std::cout << "is-ctx-dep = ";
      for (size_t i = 0;i < is_ctx_dep.size();i++)
        WriteBasicType(std::cout, false, static_cast<bool>(is_ctx_dep[i]));
      std::cout << "hmm_lengths = "; WriteIntegerVector(std::cout, false, hmm_lengths);
      std::cout << "phone_ids = "; WriteIntegerVector(std::cout, false, phone_ids);
      std::cout << "Stats are: \n";
      WriteBuildTreeStats(std::cout, false, stats);
    }

    // Now build the tree.

    Questions qopts;
    int32 num_quest = rand() % 10, num_iters = rand () % 5;
    qopts.InitRand(stats, num_quest, num_iters, kAllKeysUnion);  // This was tested in build-tree-utils-test.cc

    {
      std::cout << "Printing questions:\n";
      std::vector<EventKeyType> keys;
      qopts.GetKeysWithQuestions(&keys);
      for (size_t i = 0;i < keys.size();i++) {
        KALDI_ASSERT(qopts.HasQuestionsForKey(keys[i]));
        const QuestionsForKey &opts = qopts.GetQuestionsOf(keys[i]);
        std::cout << "num-quest: "<< opts.initial_questions.size() << '\n';
        for (size_t j = 0;j < opts.initial_questions.size();j++) {
          for (size_t k = 0;k < opts.initial_questions[j].size();k++)
            std::cout << opts.initial_questions[j][k] <<" ";
          std::cout << '\n';
        }
      }
    }

    float thresh = 100.0 * RandUniform();
    int max_leaves = 100;
    std::cout <<"Thresh = "<<thresh<<" for building tree.\n";

    {
      std::cout << "Building tree\n";
      EventMap *tree = NULL;
      std::vector<std::vector<int32> > phone_sets(phone_ids.size());
      for (size_t i = 0; i < phone_ids.size(); i++)
        phone_sets[i].push_back(phone_ids[i]);
      std::vector<bool> share_roots(phone_sets.size(), true),
          do_split(phone_sets.size(), true);

      tree = BuildTree(qopts, phone_sets, hmm_lengths, share_roots,
                       do_split, stats, thresh, max_leaves, 0.0, P);
      // Would have print-out & testing code here.
      std::cout << "Tree [default build] is:\n";
      tree->Write(std::cout, false);
      delete tree;
    }
    DeleteBuildTreeStats(&stats);
  }
}


} // end namespace kaldi

int main() {
  kaldi::TestGenRandStats();
  kaldi::TestBuildTree();
}

