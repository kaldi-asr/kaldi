// ivector/agglomerative-clustering.cc

// Copyrigh  2017-2018  Matthew Maciejewski
//                2018  David Snyder

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
  while (nclusters_ > min_clust_ && !queue_.empty()) {
    std::pair<BaseFloat, std::pair<uint16, uint16> > pr = queue_.top();
    int32 i = (int32) pr.second.first, j = (int32) pr.second.second;
    queue_.pop();
    if ((active_clusters_.find(i) != active_clusters_.end()) &&
        (active_clusters_.find(j) != active_clusters_.end()))
      MergeClusters(i, j);
  }

  std::vector<int32> new_assignments(npoints_);
  int32 i = 0;
  std::set<int32>::iterator it;
  for (it = active_clusters_.begin(); it != active_clusters_.end(); ++it) {
    ++i;
    AhcCluster *cluster = clusters_map_[*it];
    std::vector<int32>::iterator utt_it;
    for (utt_it = cluster->utt_ids.begin();
         utt_it != cluster->utt_ids.end(); ++utt_it)
      new_assignments[*utt_it] = i;
    delete cluster;
  }
  assignments_->swap(new_assignments);
}

BaseFloat AgglomerativeClusterer::ScoreLookup(int32 i, int32 j) {
  if (i < j)
    return cluster_score_map_[std::make_pair(i, j)];
  else
    return cluster_score_map_[std::make_pair(j, i)];
}

void AgglomerativeClusterer::Initialize() {
  KALDI_ASSERT(nclusters_ != 0);
  for (int32 i = 0; i < nclusters_; i++) {
    std::vector<int32> ids;
    ids.push_back(i);
    AhcCluster *c = new AhcCluster(++ct_, -1, -1, ids);
    clusters_map_[ct_] = c;
    active_clusters_.insert(ct_);

    for (int32 j = i+1; j < nclusters_; j++) {
      BaseFloat score = scores_(i,j);
      cluster_score_map_[std::make_pair(i+1, j+1)] = score;
      if (score <= thresh_)
        queue_.push(std::make_pair(score,
            std::make_pair(static_cast<uint16>(i+1),
                           static_cast<uint16>(j+1))));
    }
  }
}

void AgglomerativeClusterer::MergeClusters(int32 i, int32 j) {
  AhcCluster *clust1 = clusters_map_[i];
  AhcCluster *clust2 = clusters_map_[j];
  clust1->id = ++ct_;
  clust1->parent1 = i;
  clust1->parent2 = j;
  clust1->size += clust2->size;
  clust1->utt_ids.insert(clust1->utt_ids.end(), clust2->utt_ids.begin(),
                         clust2->utt_ids.end());
  active_clusters_.erase(i);
  active_clusters_.erase(j);
  std::set<int32>::iterator it;
  for (it = active_clusters_.begin(); it != active_clusters_.end(); ++it) {
    BaseFloat new_score = ScoreLookup(*it, i) + ScoreLookup(*it, j);
    cluster_score_map_[std::make_pair(*it, ct_)] = new_score;
    BaseFloat norm = clust1->size * (clusters_map_[*it])->size;
    if (new_score / norm <= thresh_)
      queue_.push(std::make_pair(new_score / norm,
          std::make_pair(static_cast<uint16>(*it),
                         static_cast<uint16>(ct_))));
  }
  active_clusters_.insert(ct_);
  clusters_map_[ct_] = clust1;
  delete clust2;
  nclusters_--;
}

void AgglomerativeCluster(
    const Matrix<BaseFloat> &scores,
    BaseFloat thresh,
    int32 min_clust,
    std::vector<int32> *assignments_out) {
  KALDI_ASSERT(min_clust >= 0);
  AgglomerativeClusterer ac(scores, thresh, min_clust, assignments_out);
  ac.Cluster();
}

}  // end namespace kaldi.
