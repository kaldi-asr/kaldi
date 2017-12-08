// ivector/agglomerative-bottom-up-clustering.cc

// Copyright 2012   Arnab Ghoshal
//           2009-2011  Microsoft Corporation;  Saarland University
//           2016  David Snyder
//           2017  Matthew Maciejewski

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
#include <queue>

#include "ivector/agglomerative-bottom-up-clustering.h"

namespace kaldi {

typedef uint16 uint_smaller;
typedef int16 int_smaller; 

struct AHCCluster {
  int32 id, parent1, parent2, size;
  std::vector<int32> utt_ids;
  AHCCluster(int32 id, int32 p1, int32 p2, std::vector<int32> utts)
      : id(id), parent1(p1), parent2(p2), utt_ids(utts) {
    size = utts.size();
  }
};

class AgglomerativeClusterer {
 public:
  AgglomerativeClusterer(
      const std::vector<std::string> &uttlist,
      const std::unordered_map<std::string, BaseFloat> &score_map,
      BaseFloat max_dist,
      BaseFloat thresh,
      int32 min_clust,
      std::vector<int32> *assignments_out)
      : ct_(0), uttlist_(uttlist), score_map_(score_map),
        max_dist_(max_dist), thresh_(thresh), min_clust_(min_clust),
        assignments_(assignments_out != NULL ?
                         assignments_out : &tmp_assignments_) {
    nclusters_ = npoints_ = uttlist.size();
  }

  void Cluster();

 private:
  BaseFloat ScoreLookup(int32 i, int32 j);
  void Initialize();
  void MergeClusters(int32 i, int32 j);

  int32 ct_;
  const std::vector<std::string> &uttlist_;
  const std::unordered_map<std::string, BaseFloat> &score_map_;
  BaseFloat max_dist_;
  BaseFloat thresh_;
  int32 min_clust_;
  std::vector<int32> *assignments_;

  std::vector<int32> tmp_assignments_;

  std::unordered_map<std::pair<int32, int32>, BaseFloat,
                     PairHasher<int32, int32>> cluster_score_map_;
  std::unordered_map<int32, AHCCluster*> clusters_map_;
  std::set<int32> active_clusters_;
  int32 nclusters_;
  int32 npoints_;
  typedef std::pair<BaseFloat, std::pair<uint_smaller, uint_smaller> > QueueElement;
  // Priority queue using greater (lowest distances are highest priority).
  typedef std::priority_queue<QueueElement, std::vector<QueueElement>,
      std::greater<QueueElement>  > QueueType;
  QueueType queue_;
};

void AgglomerativeClusterer::Cluster() {
  KALDI_VLOG(2) << "Initializing cluster assignments.";
  Initialize();

  KALDI_VLOG(2) << "Clustering...";
  while (nclusters_ > min_clust_ && !queue_.empty()) {
    std::pair<BaseFloat, std::pair<uint_smaller, uint_smaller> > pr = queue_.top();
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
    AHCCluster* cluster = clusters_map_[*it];
    std::vector<int32>::iterator utt_it;
    for (utt_it = cluster->utt_ids.begin();
         utt_it != cluster->utt_ids.end(); ++utt_it)
      new_assignments[*utt_it] = i;
    delete(cluster);
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
    AHCCluster* c = new AHCCluster(++ct_, -1, -1, ids);
    clusters_map_[ct_] = c;
    active_clusters_.insert(ct_);

    for (int32 j = i+1; j < nclusters_; j++) {
      std::string utts_key;
      if (uttlist_[i] < uttlist_[j])
        utts_key = uttlist_[i]+uttlist_[j];
      else
        utts_key = uttlist_[j]+uttlist_[i];
      BaseFloat score;
      std::unordered_map<std::string, BaseFloat>::const_iterator it =
          score_map_.find(utts_key);
      if (it == score_map_.end())
        score = max_dist_;
      else
        score = it->second;

      cluster_score_map_[std::make_pair(i+1, j+1)] = score;
      if (score <= thresh_)
        queue_.push(std::make_pair(score,
            std::make_pair(static_cast<uint_smaller>(i+1),
                           static_cast<uint_smaller>(j+1))));
    }
  }
}

void AgglomerativeClusterer::MergeClusters(int32 i, int32 j) {
  AHCCluster* clust1 = clusters_map_[i];
  AHCCluster* clust2 = clusters_map_[j];
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
          std::make_pair(static_cast<uint_smaller>(*it),
                         static_cast<uint_smaller>(ct_))));
  }
  active_clusters_.insert(ct_);
  clusters_map_[ct_] = clust1;
  delete(clust2);
  nclusters_--;
}

void AgglomerativeClusterBottomUp(
    const std::vector<std::string> &uttlist,
    const std::unordered_map<std::string, BaseFloat> &score_map,
    BaseFloat max_dist,
    BaseFloat thresh,
    int32 min_clust,
    std::vector<int32> *assignments_out) {
  KALDI_ASSERT(thresh >= 0.0 && min_clust >= 0);
  AgglomerativeClusterer ac(uttlist, score_map, max_dist, thresh,
                            min_clust, assignments_out);
  ac.Cluster();
}

}  // end namespace kaldi.
