// tree/build-tree-utils-multi.cc

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
#include <vector>
#include "util/stl-utils.h"
#include "tree/build-tree-utils.h"
#include "tree/clusterable-classes.h"

/*
  MultiDecisionTreeBuilder is a class used in SplitDecisionTreeMulti
*/
namespace kaldi {

using std::vector;

class MultiDecisionTreeSplitter {
 public:
  EventMap *GetMap() {
    if (!yes_) {  // leaf.
      return new ConstantEventMap(leaf_);
    } else {
      return new SplitEventMap(key_, yes_set_, yes_->GetMap(), no_->GetMap());
    }
  }
  BaseFloat BestSplit() { return best_split_impr_; }
  // returns objf improvement (>=0) of best possible split.

  void DoSplit(int32 *next_leaf) {
    if (!yes_) {  // not already split; we are a leaf, so split.
      DoSplitInternal(next_leaf);
    } else {  // find which of our children is best to split, and split that.
      (yes_->BestSplit() >= no_->BestSplit() ? yes_ : no_)->DoSplit(next_leaf);
      best_split_impr_ = std::max(yes_->BestSplit(),
                                  no_->BestSplit());  // may have changed.
    }
  }

  void SetTreeIndex(int32 i) {
    tree_index_ = i;
  }

  // this should only be called once, after SplitStatsByMap
  void SetLeafValue(int32 j) {
    for (int32 i = 0; i < stats_.size(); i++) {
      (dynamic_cast<EntropyClusterable*>(stats_[i].second))
        ->SetLeafValue(tree_index_, j);
    }
  }

  // this should only be called **before** the splitting begins
  void SetNumTrees(int32 num_trees) {
    num_trees_ = num_trees;
    for (int32 i = 0; i<stats_.size(); i++) {
      (dynamic_cast<EntropyClusterable*>(stats_[i].second))
        ->SetNumTrees(num_trees);
    }
  }

  BaseFloat RecomputeBestSplit() {
    if (!yes_) { // this is a leaf
      FindBestSplit();
      return best_split_impr_;
    } else {
      MultiDecisionTreeSplitter *best =
           (yes_->best_split_impr_ > no_->best_split_impr_ ? yes_ : no_);
      best->RecomputeBestSplit();
      MultiDecisionTreeSplitter *new_best =
           (yes_->best_split_impr_ > no_->best_split_impr_ ? yes_ : no_);
      if (new_best != best) { 
        new_best->RecomputeBestSplit();
      }
      best_split_impr_ =
             std::max(yes_->best_split_impr_, no_->best_split_impr_);
      return best_split_impr_;
    }
  }

  MultiDecisionTreeSplitter(EventAnswerType leaf,
                            const BuildTreeStatsType &stats,
                            const Questions &q_opts, int32 num_trees):
                      q_opts_(q_opts), 
                      yes_(NULL), no_(NULL), leaf_(leaf), stats_(stats), 
                      num_trees_(num_trees) {
    SetNumTrees(num_trees);
    FindBestSplit();
  }

  ~MultiDecisionTreeSplitter() {
    if (yes_) delete yes_;
    if (no_) delete no_;
  }

 private:
  void DoSplitInternal(int32 *next_leaf) {
    // Does the split; applicable only to leaf nodes.
    KALDI_ASSERT(!yes_);  // make sure children not already set up.
    KALDI_ASSERT(best_split_impr_ > 0);
    EventAnswerType yes_leaf = leaf_, no_leaf = (*next_leaf)++;
    leaf_ = -1;  // we now have no leaf.
    // Now split the stats.
    BuildTreeStatsType yes_stats, no_stats;
    yes_stats.reserve(stats_.size()); no_stats.reserve(stats_.size());

    for (BuildTreeStatsType::const_iterator iter = stats_.begin();
         iter != stats_.end(); ++iter) {
      const EventType &vec = iter->first;
      EventValueType val;
      if (!EventMap::Lookup(vec, key_, &val)) {
        KALDI_ERR << "DoSplitInternal: key has no value.";
      }
      if (std::binary_search(yes_set_.begin(), yes_set_.end(), val)) {
        yes_stats.push_back(*iter);
//        if (yes_leaf == 0) yes_leaf = 9999;
        dynamic_cast<EntropyClusterable*>(yes_stats[yes_stats.size()-1].second)
          ->SetLeafValue(tree_index_, yes_leaf);
        KALDI_ASSERT(dynamic_cast<EntropyClusterable*>
          (yes_stats[yes_stats.size()-1].second)->LeafCombinationCount() == 1);
      }
      else {
        no_stats.push_back(*iter);
        dynamic_cast<EntropyClusterable*>(no_stats[no_stats.size()-1].second)
          ->SetLeafValue(tree_index_, no_leaf);
        KALDI_ASSERT(dynamic_cast<EntropyClusterable*>
            (no_stats[no_stats.size()-1].second)->LeafCombinationCount() == 1);
      }
    }
#ifdef KALDI_PARANOID
    {  // Check objf improvement.
      Clusterable *yes_clust = SumStats(yes_stats),
                  *no_clust = SumStats(no_stats);

      EntropyClusterable *entro;
      entro = dynamic_cast<EntropyClusterable*>(yes_clust);
      KALDI_ASSERT(entro->LeafCombinationCount() == 1);

      entro = dynamic_cast<EntropyClusterable*>(no_clust);
      KALDI_ASSERT(entro->LeafCombinationCount() == 1);

      BaseFloat impr_check = yes_clust->Distance(*no_clust);
      // this is a negated objf improvement from merging
      // (== objf improvement from splitting).
      if (!ApproxEqual(impr_check, best_split_impr_, 0.01)) {
        KALDI_WARN << "DoSplitInternal: possible problem: "
                   << impr_check << " != " << best_split_impr_;
      }
      delete yes_clust; delete no_clust;
    }
#endif
    yes_ =
      new MultiDecisionTreeSplitter(yes_leaf, yes_stats, q_opts_, num_trees_);
    no_ =
      new MultiDecisionTreeSplitter(no_leaf, no_stats, q_opts_, num_trees_);

    yes_->SetTreeIndex(tree_index_);
    no_->SetTreeIndex(tree_index_);

    best_split_impr_ = std::max(yes_->BestSplit(), no_->BestSplit());
    stats_.clear();  // note: pointers in stats_ were not owned here.
  }

  void FindBestSplit() {
    // This sets best_split_impr_, key_ and yes_set_.
    // May just pick best question, or may iterate a bit (depends on
    // q_opts; see FindBestSplitForKey for details)
    std::vector<EventKeyType> all_keys;
    q_opts_.GetKeysWithQuestions(&all_keys);
    if (all_keys.size() == 0) {
      KALDI_WARN << "MultiDecisionTreeSplitter::FindBestSplit(), "
                 << "no keys available to split on (maybe no key covered all "
                 << "of your events, or there was"
                 << " a problem with your questions configuration?)";
    }
    best_split_impr_ = 0;
    for (size_t i = 0;i < all_keys.size(); i++) {
      if (q_opts_.HasQuestionsForKey(all_keys[i])) {
        std::vector<EventValueType> temp_yes_set;
        BaseFloat split_improvement = FindBestSplitForKey(stats_, q_opts_,
                                                  all_keys[i], &temp_yes_set);
        if (split_improvement > best_split_impr_) {
          best_split_impr_ = split_improvement;
          yes_set_ = temp_yes_set;
          key_ = all_keys[i];
        }
      }
    }
  }

  // Data members... Always used:
  const Questions &q_opts_;
  BaseFloat best_split_impr_;

  // If already split:
  MultiDecisionTreeSplitter *yes_;
  MultiDecisionTreeSplitter *no_;

  // Otherwise:
  EventAnswerType leaf_;
  BuildTreeStatsType stats_;  // pointers inside there not owned here.

  // key and "yes set" of best split:
  EventKeyType key_;
  std::vector<EventValueType> yes_set_;

  int32 tree_index_;
  int32 num_trees_; 
};

vector<EventMap*> SplitDecisionTreeMulti(vector<EventMap*> input_map_vec, 
                            const BuildTreeStatsType &stats,
                            Questions &q_opts,
                            BaseFloat thresh,
                            int32 max_leaves,  // max_leaves<=0 -> no maximum.
                            vector<int32> *num_leaves_vec,
                            vector<BaseFloat> *obj_impr_out_vec,
                            vector<BaseFloat> *smallest_split_change_out_vec) {
  size_t num_trees = num_leaves_vec->size();
  KALDI_ASSERT(input_map_vec.size() == num_trees );
  if (smallest_split_change_out_vec != NULL) {
    KALDI_ASSERT(smallest_split_change_out_vec->size() == num_trees);
    for (size_t i = 0; i < num_trees; i++) {
      (*smallest_split_change_out_vec)[i] = 1.0e+20;
    }
  }
  if (obj_impr_out_vec != NULL) {
    KALDI_ASSERT(obj_impr_out_vec->size() == num_trees);
    for (size_t i = 0; i < num_trees; i++) {
      (*obj_impr_out_vec)[i] = 0.0;
    }
  }
  for (size_t i=0; i < num_trees; i++) {
    KALDI_ASSERT((*num_leaves_vec)[i] > 0);
    // can't be 0 or input_map would be empty.
  }
  int32 num_empty_leaves = 0;
  std::vector<std::vector<MultiDecisionTreeSplitter*> > builders_vec;
  builders_vec.resize(num_trees);
  {  // set up "builders" [one for each current leaf for each tree].
     //  This array is never extended.
     // the structures generated during splitting
     // remain as trees at each array location.
    for (size_t j = 0; j < num_trees; j++) {
      std::vector<BuildTreeStatsType> split_stats;
      SplitStatsByMap(stats, *input_map_vec[j], &split_stats);
      KALDI_ASSERT(split_stats.size() != 0);
      builders_vec[j].resize(split_stats.size());
      // size == #leaves of this particular tree
      for (size_t i = 0; i < split_stats.size();i++) {
        EventAnswerType leaf = static_cast<EventAnswerType>(i);
        if (split_stats[i].size() == 0) {
          num_empty_leaves++;
        }
        builders_vec[j][i] = new MultiDecisionTreeSplitter(leaf,
                                                           split_stats[i],
                                                           q_opts,
                                                           num_trees);
        builders_vec[j][i]->SetNumTrees(num_trees);
        builders_vec[j][i]->SetTreeIndex(j);
        builders_vec[j][i]->SetLeafValue(i);
      }
    }
  }

  for (size_t i = 0; i < stats.size(); i++) {
    int32 num_trees_actual =
         (dynamic_cast<EntropyClusterable*>(stats[i].second))->GetNumTrees();
    KALDI_ASSERT(num_trees_actual == num_trees);
  }

  {  // Do the splitting.
    int32 count = 0; // to keep track of how many times we have splitted
    std::priority_queue<std::pair<BaseFloat, 
                        std::pair<size_t, size_t> > > queue;
                                             //       tree_index, leaf_number
    // Initialize queue.
    for (size_t j = 0; j < num_trees; j++) {
      for (size_t i = 0;i < builders_vec[j].size(); i++) {
        queue.push(std::make_pair(builders_vec[j][i]->BestSplit(),
                                  std::make_pair(j,i)));
      } 
    }
 
    // Note-- queue's size never increases from now.
    // All the alternatives leaves to split are
    // inside the "DecisionTreeSplitter*" objects, in a tree structure.
    while (queue.top().first > thresh) {
    // queue.top(().first here might not be accurate,
    // but just an "upper bound" of improvement in likelihood
      if(max_leaves > 0 &&
            (*num_leaves_vec)[queue.top().second.first] >= max_leaves) {
        // for a given tree it exceeds the max-number of leaves
        queue.pop();
        if(queue.empty()) {
          break;
        }
        else {
          continue;  // other trees might still be able to build
        }
      }

      size_t i, j;
      // j is index of tree; i index of leaf to split
      j = queue.top().second.first; 
      i = queue.top().second.second;
      
      BaseFloat real_impr = builders_vec[j][i]->RecomputeBestSplit();

      if (real_impr < thresh) {  // this is possible since real_impr is
        // generally smaller than queue.top().first
        break;
      }

      queue.pop();  // after this, top() would be an upperbound for "2nd best"
      
      if ((queue.empty()) || (real_impr >= queue.top().first)) {
      // either queue is empty or real_impr is actually the best split
        builders_vec[j][i]->DoSplit(&(*num_leaves_vec)[j]);
        queue.push(std::make_pair(builders_vec[j][i]->BestSplit(),
                                  std::make_pair(j,i))); 
        if (smallest_split_change_out_vec != NULL) {
          (*smallest_split_change_out_vec)[j] =
                   std::min((*smallest_split_change_out_vec)[j], real_impr);
        }
        if (obj_impr_out_vec != NULL) {
          (*obj_impr_out_vec)[j] += real_impr;
        }
        count++;  // split count
      } else {
        queue.push(std::make_pair(real_impr, std::make_pair(j,i)));
        // pushing in the "real" improvement
      }
    }
  }

  vector<EventMap*> answer_vec;
  answer_vec.resize(num_trees);
  for (size_t j = 0; j < num_trees; j++)
  {  // Create the output EventMap.
    std::vector<EventMap*> sub_trees(builders_vec[j].size());
    for (size_t i = 0; i < sub_trees.size(); i++) {
      sub_trees[i] = builders_vec[j][i]->GetMap();
    }
    answer_vec[j] = input_map_vec[j]->Copy(sub_trees);
    for (size_t i = 0; i < sub_trees.size(); i++) {
      delete sub_trees[i];
    }
  }
  // Free up memory.
  for(size_t j = 0; j < num_trees; j++) {
    for (size_t i = 0;i < builders_vec[j].size();i++) {
      delete builders_vec[j][i];
    }
  }
  return answer_vec;
}

void  MergeAccordingToAssignment(BuildTreeStatsType &stats,
                                 std::vector<int32> mapping,
                                 int32 tree_index) {
  for (size_t i = 0; i < stats.size(); i++) {
    dynamic_cast<EntropyClusterable*>(stats[i].second)
      ->ChangeLeafWithMapping(tree_index, mapping);
  }
}

int ClusterEventMapGetMappingEntropy(const EventMap &e_in,
                                     BuildTreeStatsType &stats,
                               // stats contain information of leaf mappings
                               // so it's not const here
                                     BaseFloat thresh,
                                     std::vector<EventMap*> *mapping,
                                     size_t tree_index) {
  // First map stats
  KALDI_ASSERT(stats.size() != 0);
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, e_in, &split_stats);
  std::vector<Clusterable*> summed_stats;
  SumStatsVec(split_stats, &summed_stats);

  std::vector<int32> indexes;
  std::vector<Clusterable*> summed_stats_contiguous;
  size_t max_index = 0;
  for (size_t i = 0;i < summed_stats.size();i++) {
    if (summed_stats[i] != NULL) {
      indexes.push_back(i);
      summed_stats_contiguous.push_back(summed_stats[i]);
      if (i > max_index) max_index = i;
    }
  }
  if (summed_stats_contiguous.empty()) {
    KALDI_WARN << "ClusterBottomUp: nothing to cluster.\n";
    return 0;  // nothing merged.
  }

  std::vector<int32> assignments;
  BaseFloat normalizer =
    SumClusterableNormalizer(summed_stats_contiguous), change;

  // this algorithm is quadratic, so might be quite slow.
  change = ClusterBottomUpEntropy(summed_stats_contiguous,
                                 thresh,
                                 0,  // no min-clust: use threshold for now.
                                 NULL,  // don't need clusters out.
                                 &assignments,
                                 tree_index);


  KALDI_ASSERT(assignments.size() == summed_stats_contiguous.size() 
               && !assignments.empty());
  size_t num_clust = 
    *std::max_element(assignments.begin(), assignments.end()) + 1;
  int32 num_combined = summed_stats_contiguous.size() - num_clust;
  KALDI_ASSERT(num_combined >= 0);

  KALDI_VLOG(2) <<  "ClusterBottomUp combined "<< num_combined 
                << " leaves and gave a likelihood change of " << change 
                << ", normalized = " << (change/normalizer) 
                << ", normalizer = " << normalizer;
  KALDI_ASSERT(change < 0.0001);  // should be negative or zero.

  KALDI_ASSERT(mapping != NULL);
  if (max_index >= mapping->size()) mapping->resize(max_index+1, NULL);

  std::vector<int32> mapping_int(max_index+1, -1);

  for (size_t i = 0;i < summed_stats_contiguous.size();i++) {
    size_t index = indexes[i];
    size_t new_index = indexes[assignments[i]];
    // index assigned by clusterig-- map to existing indices in the map,
    // that we clustered from, so we don't 
    // conflict with indices in other parts of the tree.

    KALDI_ASSERT((*mapping)[index] == NULL 
        || "Error: Cluster seems to have been called for different parts"
       " of the tree with overlapping sets of indices.");
    (*mapping)[index] = new ConstantEventMap(new_index);
    mapping_int[index] = new_index;
  }

  MergeAccordingToAssignment(stats, mapping_int, tree_index);
  DeletePointers(&summed_stats);
  return num_combined;
}

// only needed for EntropyClusterable
EventMap *ClusterEventMapRestrictedByMapEntropy(const EventMap &e_in,
                                         BuildTreeStatsType &stats,
                                         BaseFloat thresh,
                                         const EventMap &e_restrict,
                                         int32 *num_removed_ptr,
                                         size_t tree_index) {
  std::vector<EventMap*> leaf_mapping;

  std::vector<BuildTreeStatsType> split_stats;
  int num_removed = 0;
  SplitStatsByMap(stats, e_restrict, &split_stats);
  for (size_t i = 0; i < split_stats.size(); i++) {
    if (!split_stats[i].empty())
      num_removed += ClusterEventMapGetMappingEntropy(e_in, split_stats[i],
                                        thresh, &leaf_mapping, tree_index);
  }

  if (num_removed_ptr != NULL) *num_removed_ptr = num_removed;

  EventMap *ans = e_in.Copy(leaf_mapping);
  DeletePointers(&leaf_mapping);
  return ans;
}

} // end of namespace kaldi
