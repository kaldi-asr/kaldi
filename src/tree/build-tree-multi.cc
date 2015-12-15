// tree/build-tree-multi.cc

// Copyright 2015 Hainan Xu

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

#include <set>
#include <queue>
#include "util/stl-utils.h"
#include "tree/build-tree-utils.h"
#include "tree/clusterable-classes.h"

namespace kaldi {

using std::vector;

void StatsToEntropy(const BuildTreeStatsType &stats_in,
                    BuildTreeStatsType &stats_multi, double lambda) {
    stats_multi = stats_in;
    for (size_t i = 0; i < stats_in.size(); i++) {
      EntropyClusterable* entropy_stats = new EntropyClusterable();
      entropy_stats->CopyFromGaussian(stats_in[i].second);
      entropy_stats->SetLambda(lambda);
      stats_multi[i].second = entropy_stats;
    }
}

std::vector<EventMap*> BuildTreeMulti(Questions &qopts,
                           const std::vector<std::vector<int32> > &phone_sets,
                           const std::vector<int32> &phone2num_pdf_classes,
                           const std::vector<bool> &share_roots,
                           const std::vector<bool> &do_split,

                           // this is GaussianClusterables
                           const BuildTreeStatsType &original_stats,
                           BaseFloat thresh,
                           int32 max_leaves,
                           BaseFloat cluster_thresh,
                           int32 P,
                           size_t num_trees,
                           BaseFloat lambda) {
  BuildTreeStatsType stats;
  StatsToEntropy(original_stats, stats, lambda);

  KALDI_ASSERT(thresh > 0 || max_leaves > 0);
  KALDI_ASSERT(stats.size() != 0);
  KALDI_ASSERT(!phone_sets.empty()
         && phone_sets.size() == share_roots.size()
         && do_split.size() == phone_sets.size());

  // the inputs will be further checked in GetStubMap.
  int32 num_leaves = 0;
  vector<int32> num_leaves_vec;  // allocator for leaves.
  num_leaves_vec.resize(num_trees);

  vector<EventMap*> input_map_vec;
  input_map_vec.resize(num_trees);
  for (size_t i = 0; i < num_trees; i++) {
    num_leaves = 0;
    input_map_vec[i] = GetStubMap(P,
                                  phone_sets,
                                  phone2num_pdf_classes,
                                  share_roots, 
                                  &num_leaves);
    num_leaves_vec[i] = num_leaves;
  }
  EventMap* stub_map = GetStubMap(P,
                                  phone_sets,
                                  phone2num_pdf_classes,
                                  share_roots, 
                                  &num_leaves);  
  // this is purely for checking

  vector<BuildTreeStatsType> split_stats_entropy_2;
  {
    SplitStatsByMap(stats, *stub_map, &split_stats_entropy_2);
    for (size_t j = 0; j < split_stats_entropy_2.size(); j++) {
      for (size_t k = 0; k < split_stats_entropy_2[j].size(); k++) {
        dynamic_cast<EntropyClusterable*>(split_stats_entropy_2[j][k].second)
          ->SetNumTrees(num_trees);
        dynamic_cast<EntropyClusterable*>(split_stats_entropy_2[j][k].second)
          ->SetLeafValue(0, j);  // shoud have a loop too for tree-index
        KALDI_ASSERT(dynamic_cast<EntropyClusterable*>(
            split_stats_entropy_2[j][k].second)->LeafCombinationCount() == 1); 
      }

      // HERE

      EntropyClusterable *sum_entropy
       = dynamic_cast<EntropyClusterable*>(SumStats(split_stats_entropy_2[j]));
    
      if (sum_entropy == NULL) continue;  // sometimes there is no stats

      if (num_trees == 1) {
        int32 s = sum_entropy->LeafCombinationCount();
        // shouldn't be > 1 when single tree
        KALDI_ASSERT(s == 1);
      }
      delete sum_entropy;
    }
  }

  vector<BaseFloat> impr_vec;
  impr_vec.resize(num_trees);

  vector<BaseFloat> smallest_split_vec;
  smallest_split_vec.resize(num_trees);
  for (size_t i = 0; i < num_trees; i++) {
    smallest_split_vec[i] = 1.0e+10;
  }

  std::vector<int32> nonsplit_phones;
  for (size_t i = 0; i < phone_sets.size(); i++) {
    if (!do_split[i]) {
      nonsplit_phones.insert(nonsplit_phones.end(),
                    phone_sets[i].begin(), phone_sets[i].end());
    }
  }

  std::sort(nonsplit_phones.begin(), nonsplit_phones.end());

  KALDI_ASSERT(IsSortedAndUniq(nonsplit_phones));
  BuildTreeStatsType filtered_stats;
  FilterStatsByKey(stats, P, nonsplit_phones, false, &filtered_stats);
  vector<EventMap*> tree_split_vec;
  tree_split_vec.resize(num_trees);
  

  tree_split_vec = SplitDecisionTreeMulti(input_map_vec,
                                          filtered_stats,
                                          qopts, thresh, max_leaves,
                                          &num_leaves_vec, &impr_vec, 
                                          &smallest_split_vec);
  KALDI_LOG << "Split Multi done";
  BaseFloat impr = 0.0;
  
  KALDI_LOG << "multi tree build: increase in objf is ";

  for (size_t i = 0; i < impr_vec.size(); i++) {
    impr += impr_vec[i];
    KALDI_LOG << impr_vec[i] << " for tree " << i;
  }

  BaseFloat normalizer = SumNormalizer(stats),
       impr_normalized = impr / normalizer,
       normalizer_filt = SumNormalizer(filtered_stats),
       impr_normalized_filt = impr / normalizer_filt;

  std::ostringstream ostr;

  for (size_t i = 0; i < num_trees; i++) {
    ostr << "tree_" << i << ": " << num_leaves_vec[i] << ", ";
  }

  KALDI_VLOG(1) << "After decision tree split, num-leaves = " << ostr.str();

  KALDI_VLOG(1) << "Total objf-impr = " << impr_normalized << " per frame over "
                << normalizer << " frames.";
 
  KALDI_VLOG(1) << "Including just phones that were split, improvement is " 
                << impr_normalized_filt << " per frame over "
                << normalizer_filt << " frames.";
  
  EntropyClusterable *sum_of_all_entropy_stats = 
          dynamic_cast<EntropyClusterable*>(SumStats(filtered_stats)); 

  BaseFloat total_entropy = sum_of_all_entropy_stats->GetEntropy(-1);

  KALDI_VLOG(1) << "The total entropy is "
                << total_entropy;
                  // -1 is the total joint distribution's entropy

  vector<BaseFloat> entropy_vec;

  for (size_t i = 0; i < num_trees; i++) {
    BaseFloat entropy = sum_of_all_entropy_stats->GetEntropy(i);
    entropy_vec.push_back(entropy);
    KALDI_VLOG(1) <<  "The entropy of the " << i << "-th tree is "
                  << entropy;
  }

  // to check we got the objf right
  BaseFloat likelihood_before = 0;  // more like likelihood increase
  BaseFloat likelihood_after = 0;  // more like likelihood increase
  BaseFloat entropy_before = 0;  // the "monophone" tree entropy
  BaseFloat entropy_after = 0;  // the "monophone" tree entropy
  vector<BuildTreeStatsType> split_stats;
  vector<BuildTreeStatsType> split_stats_entropy;

  {
    SplitStatsByMap(original_stats, *stub_map, &split_stats);
    for (size_t j = 0; j < split_stats.size(); j++) {
      GaussClusterable *sum = 
        dynamic_cast<GaussClusterable*>(SumStats(split_stats[j]));
    
      if (sum == NULL) continue;  // sometimes there is no stats

      likelihood_before += sum->Objf();
      entropy_before += (sum->count() / sum_of_all_entropy_stats->count())
        * log(sum_of_all_entropy_stats->count() / sum->count());
      delete sum;
    }
    likelihood_before *= num_trees;
  }
  split_stats.clear();

  for (size_t i = 0; i < num_trees; i++) {
    SplitStatsByMap(original_stats, *tree_split_vec[i], &split_stats);
    SplitStatsByMap(stats, *tree_split_vec[i], &split_stats_entropy);
    for (size_t j = 0; j < split_stats.size(); j++) {
      GaussClusterable *sum =
        dynamic_cast<GaussClusterable*>(SumStats(split_stats[j]));
      EntropyClusterable *sum_entropy =
        dynamic_cast<EntropyClusterable*>(SumStats(split_stats_entropy[j]));
    
      if (sum == NULL) continue;  // sometimes there is no stats

      if (num_trees == 1) {
        // shouldn't be > 1 when single tree
        KALDI_ASSERT(sum_entropy->LeafCombinationCount() == 1);
      }

      likelihood_after += sum->Objf();
      entropy_after += (sum->count() / sum_of_all_entropy_stats->count())
        * log(sum_of_all_entropy_stats->count() / sum->count());
      delete sum;
      delete sum_entropy;
    }
  }

  delete sum_of_all_entropy_stats; 

  std::vector<BaseFloat> cluster_threshes(num_trees);

  KALDI_VLOG(1) << "Total Entropy Comparison "
                << total_entropy << " VS " << entropy_after;
  BaseFloat entropy_increase = total_entropy - entropy_before;

  BaseFloat ratio = 1 - entropy_increase * lambda / impr_normalized_filt;

  KALDI_VLOG(1) << "The log-likelihood / objf ratio is "
                << ratio;

  KALDI_VLOG(1) << "The average pure log-likelihood increase on one tree is "
                << (impr_normalized - lambda * entropy_increase) / num_trees;

  KALDI_VLOG(1) << "Checking correctness of log-likelihood "
                << (likelihood_after - likelihood_before) / normalizer 
                << " VS " << (impr_normalized - lambda * entropy_increase);

  if (cluster_thresh < 0.0) {
    BaseFloat smallest_split = *std::min_element(smallest_split_vec.begin(),
                                                 smallest_split_vec.end());
    for (size_t i = 0; i < num_trees; i++) {
      cluster_threshes[i] = smallest_split * (-cluster_thresh);
      // now the cluster_thresh for all trees are the same.
      // I usee to have different thresh for different trees, while these
      // thresh depends on the smallest_split of each tree, but it seems
      // mathematically it deosn't make a whole lot of sense

      KALDI_LOG << "Setting clustering threshold for tree "
                << i << " to "
                << cluster_threshes[i];
    }
  }
  else {
    for (size_t i = 0; i < num_trees; i++) {
      cluster_threshes[i] = cluster_thresh;
    }
  }

  BaseFloat diff = 0;
  if (cluster_thresh != 0.0) {   // Cluster the tree.
    vector<EventMap*> tree_renumbered_vec;
    tree_renumbered_vec.resize(num_trees);
    for (size_t i = 0; i < num_trees; i++) {
      BaseFloat objf_before_cluster = ObjfGivenMap(stats, *tree_split_vec[i]); 

      // Now do the clustering.
      int32 num_removed = 0;

      // change the function to somethingEntropy
      EventMap *tree_clustered = 
               ClusterEventMapRestrictedByMapEntropy(*tree_split_vec[i], 
                                                      stats,
                                                      cluster_threshes[i],
                                                      *input_map_vec[i],
                                                      &num_removed,
                                                      i);
      KALDI_LOG << "BuildTree: removed "<< num_removed << " leaves.";

      int32 num_leaves = 0;
      EventMap *tree_renumbered = 
        RenumberEventMap(*tree_clustered, &num_leaves);

      BaseFloat objf_after_cluster = ObjfGivenMap(stats, *tree_renumbered);

      KALDI_VLOG(1) << "Objf change due to clustering "
                    << ((objf_after_cluster-objf_before_cluster) / normalizer)
                    << " per frame.";
      diff += ((objf_after_cluster-objf_before_cluster) / normalizer);
      KALDI_VLOG(1) << "Normalizing over only split phones, this is: "
                    << objf_after_cluster-objf_before_cluster / normalizer_filt
                    << " per frame.";
      KALDI_VLOG(1) <<  "Num-leaves is now " << num_leaves;


      delete tree_clustered;
      delete tree_split_vec[i];
      delete input_map_vec[i];
      tree_renumbered_vec[i] = tree_renumbered;
    }

    sum_of_all_entropy_stats = 
            dynamic_cast<EntropyClusterable*>(SumStats(filtered_stats)); 

    total_entropy = sum_of_all_entropy_stats->GetEntropy(-1);
                  // -1 is the total joint distribution's entropy

    KALDI_VLOG(1) << "The total entropy is "
                  << total_entropy;

    entropy_vec.resize(0);

    for (size_t i = 0; i < num_trees; i++) {
      BaseFloat entropy = sum_of_all_entropy_stats->GetEntropy(i);
      entropy_vec.push_back(entropy);
      KALDI_VLOG(1) <<  "The entropy of the " << i << "-th tree is "
                    << entropy;
    }
    BaseFloat likelihood = 0;  // more like liklihood increase
    BaseFloat initial_entropy = 0;  // the "monophone" tree entropy
    split_stats.resize(0);
    {
      SplitStatsByMap(original_stats, *stub_map, &split_stats);
      for (size_t j = 0; j < split_stats.size(); j++) {
        GaussClusterable *sum =
          dynamic_cast<GaussClusterable*>(SumStats(split_stats[j]));
        if (sum == NULL) continue;  // sometimes there is no stats
        likelihood -= sum->Objf();
        initial_entropy += (sum->count() / sum_of_all_entropy_stats->count())
          * log(sum_of_all_entropy_stats->count() / sum->count());
        delete sum;
      }
      likelihood *= num_trees;
    }

    BaseFloat count_all = sum_of_all_entropy_stats->count();

    for (size_t i = 0; i < num_trees; i++) {
      vector<BuildTreeStatsType> split_stats;
      SplitStatsByMap(original_stats, *(tree_renumbered_vec[i]), &split_stats);
      BaseFloat entropy = 0;
      for (size_t j = 0; j < split_stats.size(); j++) {

        GaussClusterable *sum =
          dynamic_cast<GaussClusterable*>(SumStats(split_stats[j]));
        entropy += sum->count() / count_all * log(count_all / sum->count());
        likelihood += sum->Objf();
        delete sum;
      }
      KALDI_VLOG(1) <<  "The entropy of the " << i << "-th tree is "
                    << entropy;
    }

    delete sum_of_all_entropy_stats; 

    entropy_increase = total_entropy - initial_entropy;

    KALDI_VLOG(1) << "Checking correctness of log-likelihood "
                  << likelihood / normalizer << " VS " 
                  << (impr_normalized + diff - lambda * entropy_increase);
    DeleteBuildTreeStats(&stats);

    delete stub_map;
    return tree_renumbered_vec; 
  } 
  else {
    for (size_t i = 0; i < num_trees; i++) {
      delete input_map_vec[i];
    }
    DeleteBuildTreeStats(&stats);
    delete stub_map;
    return tree_split_vec; 
  }
}

} // namespace kaldi
