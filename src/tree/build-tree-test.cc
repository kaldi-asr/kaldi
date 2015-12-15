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

#include "util/common-utils.h"
#include "util/stl-utils.h"
#include "tree/build-tree.h"
#include "tree/build-tree-virtual.h"

namespace kaldi {

using std::vector;

void AssertSameVectors(const vector<int32>& v1,
                       const vector<int32>& v2) {
  KALDI_ASSERT(v1.size() == v2.size());
  for (size_t i = 0; i < v1.size(); i++) {
    KALDI_ASSERT(v1[i] == v2[i]);
  }
}

void AssertSameMap(const unordered_map<int32, vector<int32> >& m,
                   const unordered_map<int32, vector<int32> >& m2) {
  for (unordered_map<int32, vector<int32> >::const_iterator iter = m.begin();
       iter != m.end(); iter++) {
    unordered_map<int32, vector<int32> >::const_iterator iter2 =
         m2.find(iter->first);
    KALDI_ASSERT(iter2 != m2.end());
//    AssertSameVectors(iter->second, iter->second); //m2[iter->first]);
    AssertSameVectors(iter->second, iter2->second);
  }
  std::cout << "The 2 maps, with size " << m.size() << " are the same\n";
}

void TestMappingWriteAndRead(const unordered_map<int32, vector<int32> >& m,
                             size_t num_trees) {
  for (size_t i = 0; i < 2; i++) {
    bool binary = (i % 2 == 1);
    std::string filename = "mapping";
    if (binary) {
      filename += "_b";
    }
    else {
      filename += "_t";
    }
    Output output(filename, binary);

    WriteMultiTreeMapping(m, output.Stream(), binary, num_trees);
    output.Close();

    unordered_map<int32, vector<int32> > m_from_file;
    std::cout << "before binary is " << binary << std::endl;
    Input input(filename, &binary);
    std::cout << "after binary is " << binary << std::endl;
    ReadMultiTreeMapping(m_from_file, input.Stream(), binary, num_trees);
    input.Close();
    AssertSameMap(m, m_from_file);
  }
}

void TestGenRandStats() {
  for (int32 p = 0; p < 2; p++) {
    int32 dim = 1 + Rand() % 40;
    int32 num_phones = 1 + Rand() % 40;
    int32 num_stats = 1 +  (Rand() % 20);
    int32 N = 2 + Rand() % 2;  // 2 or 3.
    int32 P = Rand() % N;
    float ctx_dep_prob = 0.5 + 0.5*RandUniform();
    std::vector<int32> phone_ids(num_phones);
    for (size_t i = 0;i < (size_t)num_phones;i++)
      phone_ids[i] = (i == 0 ? (Rand() % 2) : phone_ids[i-1] + 1 + (Rand()%2));
    int32 max_phone = *std::max_element(phone_ids.begin(), phone_ids.end());
    std::vector<int32> hmm_lengths(max_phone+1);
    std::vector<bool> is_ctx_dep(max_phone+1);

    for (int32 i = 0; i <= max_phone; i++) {
      hmm_lengths[i] = 1 + Rand() % 3;
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

    int32 dim = 1 + Rand() % 40;
    int32 num_phones = 1 + Rand() % 8;
    int32 num_stats = 1 + (Rand() % 15) * (Rand() % 15);  // up to 14^2 + 1 separate stats.
    int32 N = 2 + Rand() % 2;  // 2 or 3.
    int32 P = Rand() % N;
    float ctx_dep_prob = 0.5 + 0.5*RandUniform();

    std::vector<int32> phone_ids(num_phones);
    for (size_t i = 0;i < (size_t)num_phones;i++)
      phone_ids[i] = (i == 0 ? (Rand() % 2) : phone_ids[i-1] + 1 + (Rand()%2));
    int32 max_phone = *std::max_element(phone_ids.begin(), phone_ids.end());
    std::vector<int32> hmm_lengths(max_phone+1);
    std::vector<bool> is_ctx_dep(max_phone+1);

    for (int32 i = 0; i <= max_phone; i++) {
      hmm_lengths[i] = 1 + Rand() % 3;
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
    int32 num_quest = Rand() % 10, num_iters = rand () % 5;
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

void TestBuildTreeMultiAndVirtual() {
  for (int32 p = 0; p < 14; p++) {
    double lambda = 11.0;
    if (p % 2 == 0) {
      lambda = 0.0; // we want to test for when lambda==0,
       //  all trees should be identical
      //  (given that we don't do refinment during the tree-splitting)
    }

    std::cout << "test multi, p = " << p << std::endl;
    size_t num_trees = std::min(4, p+1); 
    // First decide phone-ids, hmm lengths, is-ctx-dep...

    int32 dim = 1 + rand() % 40;
    int32 num_phones = 1 + rand() % 8;
    int32 num_stats = 1 + (rand() % 1555) * (rand() % 1555);  // up to 14^2 + 1 separate stats.
    int32 N = 2 + rand() % 2;  // 2 or 3.
    int32 P = rand() % N;
    float ctx_dep_prob = 0.5 + 0.5 * RandUniform();

    std::vector<int32> phone_ids(num_phones);
    for (size_t i = 0; i < (size_t)num_phones; i++)
      phone_ids[i] = (i == 0 ? (rand() % 2) : phone_ids[i-1] + 1 + (rand()%2));
    int32 max_phone = *std::max_element(phone_ids.begin(), phone_ids.end());
    std::vector<int32> hmm_lengths(max_phone+1);
    std::vector<bool> is_ctx_dep(max_phone+1);

    for (int32 i = 0; i <= max_phone; i++) {
      hmm_lengths[i] = 1 + rand() % 3;
      is_ctx_dep[i] = (RandUniform() < ctx_dep_prob);  // true w.p. ctx_dep_prob.
    }
    for (size_t i = 0; i < (size_t) num_phones; i++) {
      KALDI_VLOG(2) << "For idx = "<< i << ", (phone_id, hmm_length, is_ctx_dep) == "
                    << (phone_ids[i]) << " " << (hmm_lengths[phone_ids[i]]) << " "
                    << (is_ctx_dep[phone_ids[i]]);
    }
    // Generate rand stats.  These were tested in TestGenRandStats() above.
    BuildTreeStatsType stats;
    bool ensure_all_covered = false;
    GenRandStats(dim, num_stats, N, P, phone_ids, hmm_lengths,
                 is_ctx_dep, ensure_all_covered, &stats);

    {  // print out the stats.
      std::cout << "multi Writing random stats.";
      std::cout << "dim = " << dim << '\n';
      std::cout << "multi num_phones = " << num_phones << '\n';
      std::cout << "multi num_stats = " << num_stats << '\n';
      std::cout << "N = "<< N << '\n';
      std::cout << "P = "<< P << '\n';
      std::cout << "is-ctx-dep = ";
      for (size_t i = 0; i < is_ctx_dep.size(); i++)
        WriteBasicType(std::cout, false, static_cast<bool>(is_ctx_dep[i]));
      std::cout << "hmm_lengths = "; WriteIntegerVector(std::cout, false, hmm_lengths);
      std::cout << "phone_ids = "; WriteIntegerVector(std::cout, false, phone_ids);
      std::cout << "Stats are: \n";
      WriteBuildTreeStats(std::cout, false, stats);
    }

    // Now build the tree.

    Questions qopts;
    int32 num_quest = rand() % 10, num_iters = rand () % 5;

    if (p % 2 == 0) {
      num_iters = 0; // to turn off refinement s.t we get identical trees
    }

    qopts.InitRand(stats, num_quest, num_iters, kAllKeysUnion);

    {
      std::cout << "Printing questions:\n";
      std::vector<EventKeyType> keys;
      qopts.GetKeysWithQuestions(&keys);
      for (size_t i = 0; i < keys.size(); i++) {
        KALDI_ASSERT(qopts.HasQuestionsForKey(keys[i]));
        const QuestionsForKey &opts = qopts.GetQuestionsOf(keys[i]);
        std::cout << "num-quest: " << opts.initial_questions.size() << '\n';
        for (size_t j = 0; j < opts.initial_questions.size(); j++) {
          for (size_t k = 0; k < opts.initial_questions[j].size(); k++)
            std::cout << opts.initial_questions[j][k] << " ";
          std::cout << '\n';
        }
      }
    }

    float thresh = 100.0 * RandUniform();
    int max_leaves = num_stats / 200;
    std::cout << "Thresh = " << thresh << " for building tree.\n";

    {
      std::cout << "Building tree\n";
      vector<EventMap*> tree_vec;
//      tree_vec.resize(num_trees);

      std::vector<std::vector<int32> > phone_sets(phone_ids.size());
      for (size_t i = 0; i < phone_ids.size(); i++)
        phone_sets[i].push_back(phone_ids[i]);
      std::vector<bool> share_roots(phone_sets.size(), true),
          do_split(phone_sets.size(), true);

      EventMap* tree_single = BuildTree(qopts, phone_sets, hmm_lengths, share_roots,
                       do_split, stats, thresh, max_leaves, 0.0, P); 
 
      tree_vec = BuildTreeMulti(qopts, phone_sets, hmm_lengths, share_roots,
                       do_split, stats, thresh, max_leaves, 0.0, P, num_trees, lambda);

      std::vector<const EventMap*> tree_vec_const;
      // just need the "vector<const EventMap*>" type

      for (size_t i = 0; i < tree_vec.size(); i++) {
        tree_vec_const.push_back(tree_vec[i]);
      }

      MultiTreePdfMap multi_pdf(tree_vec_const, N, P, hmm_lengths);

      unordered_map<int32, vector<int32> > mappings;
      EventMap* virtual_tree = multi_pdf.GenerateVirtualTree(mappings);

      if (virtual_tree == NULL) {
        KALDI_LOG << "Virtual tree failed ";
        continue;
      }


      TestMappingWriteAndRead(mappings, num_trees);


      if (p % 2 == 0) {
        if (!tree_vec[0]->IsSameTree(tree_single)) {
          KALDI_LOG << "Warning, not same with single tree ";
          KALDI_ASSERT(false);  // not sure why but it happens
        }
        for (size_t i = 0; i < num_trees; i++) { 
          if (!tree_vec[0]->IsSameTree(tree_vec[i])) {
            KALDI_LOG << "Warning, not same among multi tree ";
            KALDI_ASSERT(false);
          }
        }
        KALDI_ASSERT(virtual_tree->IsSameTree(tree_vec[0]));
      }  
      else {
        std::cout << "num_leaves, num_trees : ";
        WriteMultiTreeMapping(mappings, std::cout, false, num_trees);
        bool all_same = true;
        for (size_t i = 1; i < num_trees; i++) {
          all_same = all_same && tree_vec[0]->IsSameTree(tree_vec[i]);
        }
        if (all_same) {
          KALDI_LOG << "Warning! All trees are same. "
                       "This doesn't necessarily indicate there is a problem "
                       "but if it happens a lot then more than likely something is wrong";
        }
      }
      delete tree_single;
      // Would have print-out & testing code here.
      std::cout << "Tree [default build] is:\n";
      for (size_t i = 0; i < num_trees; i++) {
        tree_vec[i]->Write(std::cout, false);
        delete tree_vec[i];
      }
      delete virtual_tree;
    }
    DeleteBuildTreeStats(&stats);
  }
}

} // namespace kaldi

int main() {
  kaldi::TestGenRandStats();
  kaldi::TestBuildTree();
  kaldi::TestBuildTreeMultiAndVirtual();
}

