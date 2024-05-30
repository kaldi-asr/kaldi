// ivector/agglomerative-clustering.cc

// Copyright  2017-2018  Matthew Maciejewski
//                 2018  David Snyder
//                 2019  Dogan Can

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

#include <algorithm>
#include "ivector/agglomerative-clustering.h"

namespace kaldi {

void AgglomerativeClusterer::Cluster() {
  if (num_points_ > first_pass_max_points_)
    ClusterTwoPass();
  else
    ClusterSinglePass();
}

void AgglomerativeClusterer::ClusterSinglePass() {
  InitializeClusters(0, num_points_);
  ComputeClusters(min_clusters_);
  AssignClusters();
}

void AgglomerativeClusterer::ClusterTwoPass() {
  // This is the first pass loop. We divide the input into equal size subsets
  // making sure each subset has at most first_pass_max_points_ points. Then, we
  // cluster the points in each subset separately until a stopping criterion is
  // reached. We set the minimum number of clusters to 10 * min_clusters_ for
  // each subset to avoid early merging of most clusters that would otherwise be
  // kept separate in single pass clustering.
  BaseFloat num_points = static_cast<BaseFloat>(num_points_);
  int32 num_subsets = ceil(num_points / first_pass_max_points_);
  int32 subset_size = ceil(num_points / num_subsets);
  for (int32 n = 0; n < num_points_; n += subset_size) {
    InitializeClusters(n, std::min(n + subset_size, num_points_));
    ComputeClusters(min_clusters_ * 10);
    AddClustersToSecondPass();
  }

  // We swap the contents of the first and second pass data structures so that
  // we can use the same method to do second pass clustering.
  clusters_map_.swap(second_pass_clusters_map_);
  active_clusters_.swap(second_pass_active_clusters_);
  cluster_cost_map_.swap(second_pass_cluster_cost_map_);
  queue_.swap(second_pass_queue_);
  count_ = second_pass_count_;

  // This is the second pass. It moves through the queue merging clusters
  // determined in the first pass until a stopping criterion is reached.
  ComputeClusters(min_clusters_);

  AssignClusters();
}

uint32 AgglomerativeClusterer::EncodePair(int32 i, int32 j) {
  if (i < j)
    return (static_cast<uint32>(i) << 16) + static_cast<uint32>(j);
  else
    return (static_cast<uint32>(j) << 16) + static_cast<uint32>(i);
}

std::pair<int32, int32> AgglomerativeClusterer::DecodePair(uint32 key) {
  return std::make_pair(static_cast<int32>(key >> 16),
                        static_cast<int32>(key & 0x0000FFFFu));
}

void AgglomerativeClusterer::InitializeClusters(int32 first, int32 last) {
  KALDI_ASSERT(last > first);
  clusters_map_.clear();
  active_clusters_.clear();
  cluster_cost_map_.clear();
  queue_ = QueueType();  // priority_queue does not have a clear method

  for (int32 i = first; i < last; i++) {
    // create an initial cluster of size 1 for each point
    std::vector<int32> ids;
    ids.push_back(i);
    AhcCluster *c = new AhcCluster(i + 1, -1, -1, ids);
    clusters_map_[i + 1] = c;
    active_clusters_.insert(i + 1);

    // propagate the queue with all pairs from the cost matrix
    for (int32 j = i + 1; j < last; j++) {
      BaseFloat cost = costs_(i, j);
      uint32 key = EncodePair(i + 1, j + 1);
      cluster_cost_map_[key] = cost;
      if (cost <= threshold_)
        queue_.push(std::make_pair(cost, key));
    }
  }
}

void AgglomerativeClusterer::ComputeClusters(int32 min_clusters) {
  while (active_clusters_.size() > min_clusters && !queue_.empty()) {
    std::pair<BaseFloat, uint32> pr = queue_.top();
    int32 i, j;
    std::tie(i, j) = DecodePair(pr.second);
    queue_.pop();
    // check to make sure clusters have not already been merged
    if ((active_clusters_.find(i) != active_clusters_.end()) &&
        (active_clusters_.find(j) != active_clusters_.end())) {
      if (clusters_map_[i]->size + clusters_map_[j]->size <= max_cluster_size_)
        MergeClusters(i, j);
    }
  }
}

void AgglomerativeClusterer::MergeClusters(int32 i, int32 j) {
  AhcCluster *clust1 = clusters_map_[i];
  AhcCluster *clust2 = clusters_map_[j];
  // For memory efficiency, the first cluster is updated to contain the new
  // merged cluster information, and the second cluster is later deleted.
  clust1->id = ++count_;
  clust1->parent1 = i;
  clust1->parent2 = j;
  clust1->size += clust2->size;
  clust1->utt_ids.insert(clust1->utt_ids.end(), clust2->utt_ids.begin(),
                         clust2->utt_ids.end());
  // Remove the merged clusters from the list of active clusters.
  active_clusters_.erase(i);
  active_clusters_.erase(j);
  // Update the queue with all the new costs involving the new cluster
  std::set<int32>::iterator it;
  for (it = active_clusters_.begin(); it != active_clusters_.end(); ++it) {
    // The new cost is the sum of the costs of the new cluster's parents
    BaseFloat new_cost = cluster_cost_map_[EncodePair(*it, i)] +
                         cluster_cost_map_[EncodePair(*it, j)];
    uint32 new_key = EncodePair(*it, count_);
    cluster_cost_map_[new_key] = new_cost;
    BaseFloat norm = clust1->size * (clusters_map_[*it])->size;
    if (new_cost / norm <= threshold_)
      queue_.push(std::make_pair(new_cost / norm, new_key));
  }
  active_clusters_.insert(count_);
  clusters_map_[count_] = clust1;
  delete clust2;
}

void AgglomerativeClusterer::AddClustersToSecondPass() {
  // This method collects the results of first pass clustering for one subset,
  // i.e. adds the set of active clusters to the set of second pass active
  // clusters and computes the costs for the newly formed cluster pairs.
  std::set<int32>::iterator it1, it2;
  int32 count = second_pass_count_;
  for (it1 = active_clusters_.begin(); it1 != active_clusters_.end(); ++it1) {
    AhcCluster *clust1 = clusters_map_[*it1];
    second_pass_clusters_map_[++count] = clust1;

    // Compute new cluster pair costs
    for (it2 = second_pass_active_clusters_.begin();
         it2 != second_pass_active_clusters_.end(); ++it2) {
      AhcCluster *clust2 = second_pass_clusters_map_[*it2];
      uint32 new_key = EncodePair(count, *it2);

      BaseFloat new_cost = 0.0;
      std::vector<int32>::iterator utt_it1, utt_it2;
      for (utt_it1 = clust1->utt_ids.begin();
           utt_it1 != clust1->utt_ids.end(); ++utt_it1) {
         for (utt_it2 = clust2->utt_ids.begin();
              utt_it2 != clust2->utt_ids.end(); ++utt_it2) {
           new_cost += costs_(*utt_it1, *utt_it2);
         }
      }

      second_pass_cluster_cost_map_[new_key] = new_cost;
      BaseFloat norm = clust1->size * clust2->size;
      if (new_cost / norm <= threshold_)
        second_pass_queue_.push(std::make_pair(new_cost / norm, new_key));
    }

    // Copy cluster pair costs that were already computed in the first pass
    int32 count2 = second_pass_count_;
    for (it2 = active_clusters_.begin(); it2 != it1; ++it2) {
      uint32 key = EncodePair(*it1, *it2);
      BaseFloat cost = cluster_cost_map_[key];
      BaseFloat norm = clust1->size * (clusters_map_[*it2])->size;
      uint32 new_key = EncodePair(count, ++count2);
      second_pass_cluster_cost_map_[new_key] = cost;
      if (cost / norm <= threshold_)
        second_pass_queue_.push(std::make_pair(cost / norm, new_key));
    }
  }
  // We update second_pass_count_ and second_pass_active_clusters_ here since
  // above loop assumes they do not change while the loop is running.
  while (second_pass_count_ < count)
    second_pass_active_clusters_.insert(++second_pass_count_);
}

void AgglomerativeClusterer::AssignClusters() {
  assignments_->resize(num_points_);
  int32 label_id = 0;
  std::set<int32>::iterator it;
  // Iterate through the clusters and assign all utterances within the cluster
  // an ID label unique to the cluster. This is the final output and frees up
  // the cluster memory accordingly.
  for (it = active_clusters_.begin(); it != active_clusters_.end(); ++it) {
    ++label_id;
    AhcCluster *cluster = clusters_map_[*it];
    std::vector<int32>::iterator utt_it;
    for (utt_it = cluster->utt_ids.begin();
         utt_it != cluster->utt_ids.end(); ++utt_it)
      (*assignments_)[*utt_it] = label_id;
    delete cluster;
  }
}

void AgglomerativeCluster(
    const Matrix<BaseFloat> &costs,
    BaseFloat threshold,
    int32 min_clusters,
    int32 first_pass_max_points,
    BaseFloat max_cluster_fraction,
    std::vector<int32> *assignments_out) {
  KALDI_ASSERT(min_clusters >= 0);
  KALDI_ASSERT(max_cluster_fraction >= 1.0 / min_clusters);
  AgglomerativeClusterer ac(costs, threshold, min_clusters,
                            first_pass_max_points, max_cluster_fraction,
                            assignments_out);
  ac.Cluster();
}

}  // end namespace kaldi.
