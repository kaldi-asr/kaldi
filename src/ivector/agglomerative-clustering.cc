// ivector/agglomerative-clustering.cc

// Copyright  2017-2018  Matthew Maciejewski
//                 2018  David Snyder

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
  KALDI_VLOG(2) << "Initializing cluster assignments.";
  Initialize();

  KALDI_VLOG(2) << "Clustering...";
  // This is the main algorithm loop. It moves through the queue merging
  // clusters until a stopping criterion has been reached.
  while (num_clusters_ > min_clust_ && !queue_.empty()) {
    std::pair<BaseFloat, std::pair<uint16, uint16> > pr = queue_.top();
    int32 i = (int32) pr.second.first, j = (int32) pr.second.second;
    queue_.pop();
    // check to make sure clusters have not already been merged
    if ((active_clusters_.find(i) != active_clusters_.end()) &&
        (active_clusters_.find(j) != active_clusters_.end()))
      MergeClusters(i, j);
  }

  std::vector<int32> new_assignments(num_points_);
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
      new_assignments[*utt_it] = label_id;
    delete cluster;
  }
  assignments_->swap(new_assignments);
}

BaseFloat AgglomerativeClusterer::GetCost(int32 i, int32 j) {
  if (i < j)
    return cluster_cost_map_[std::make_pair(i, j)];
  else
    return cluster_cost_map_[std::make_pair(j, i)];
}

void AgglomerativeClusterer::Initialize() {
  KALDI_ASSERT(num_clusters_ != 0);
  for (int32 i = 0; i < num_points_; i++) {
    // create an initial cluster of size 1 for each point
    std::vector<int32> ids;
    ids.push_back(i);
    AhcCluster *c = new AhcCluster(++count_, -1, -1, ids);
    clusters_map_[count_] = c;
    active_clusters_.insert(count_);

    // propagate the queue with all pairs from the cost matrix
    for (int32 j = i+1; j < num_clusters_; j++) {
      BaseFloat cost = costs_(i,j);
      cluster_cost_map_[std::make_pair(i+1, j+1)] = cost;
      if (cost <= thresh_)
        queue_.push(std::make_pair(cost,
            std::make_pair(static_cast<uint16>(i+1),
                           static_cast<uint16>(j+1))));
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
    BaseFloat new_cost = GetCost(*it, i) + GetCost(*it, j);
    cluster_cost_map_[std::make_pair(*it, count_)] = new_cost;
    BaseFloat norm = clust1->size * (clusters_map_[*it])->size;
    if (new_cost / norm <= thresh_)
      queue_.push(std::make_pair(new_cost / norm,
          std::make_pair(static_cast<uint16>(*it),
                         static_cast<uint16>(count_))));
  }
  active_clusters_.insert(count_);
  clusters_map_[count_] = clust1;
  delete clust2;
  num_clusters_--;
}

void AgglomerativeCluster(
    const Matrix<BaseFloat> &costs,
    BaseFloat thresh,
    int32 min_clust,
    std::vector<int32> *assignments_out) {
  KALDI_ASSERT(min_clust >= 0);
  AgglomerativeClusterer ac(costs, thresh, min_clust, assignments_out);
  ac.Cluster();
}

}  // end namespace kaldi.
