// tree/cluster-utils.cc

// Copyright 2012   Arnab Ghoshal
// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//           2015   Hainan Xu

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

#include <functional>
#include <queue>
#include <vector>
using std::vector;

#include "base/kaldi-math.h"
#include "util/stl-utils.h"
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"

namespace kaldi {

typedef uint16 uint_smaller;
typedef int16 int_smaller;

class BottomUpClustererEntropy {
 public:
  BottomUpClustererEntropy(const std::vector<Clusterable*> &points,
                    BaseFloat max_merge_thresh,
                    int32 min_clust,
                    std::vector<Clusterable*> *clusters_out,
                    std::vector<int32> *assignments_out,
                    size_t tree_index)
      : ans_(0.0), points_(points), max_merge_thresh_(max_merge_thresh),
        min_clust_(min_clust), clusters_(clusters_out != NULL? clusters_out
            : &tmp_clusters_), assignments_(assignments_out != NULL ?
                assignments_out : &tmp_assignments_),
                tree_index_(tree_index) {

    for (size_t i = 0; i < points.size(); i++) {
      KALDI_ASSERT(
       dynamic_cast<EntropyClusterable*>(points[i])->LeafCombinationCount() == 1
       || dynamic_cast<EntropyClusterable*>(points[i])->GetNumTrees() != 1);
    }

    nclusters_ = npoints_ = points.size();
    dist_vec_.resize((npoints_ * (npoints_ - 1)) / 2);
  }

  BaseFloat Cluster();
  ~BottomUpClustererEntropy() { DeletePointers(&tmp_clusters_); }

 private:
  void Renumber();
  void InitializeAssignments();
  void SetInitialDistances();  ///< Sets up distances and queue.
  /// CanMerge returns true if i and j are existing clusters, and the distance
  /// (negated objf-change) "dist" is accurate (i.e. not outdated).
  bool CanMerge(int32 i, int32 j, BaseFloat dist);
  /// Merge j into i and delete j.
  void MergeClusters(int32 i, int32 j);
  /// Reconstructs the priority queue from the distances.
  void ReconstructQueue();

  void SetDistance(int32 i, int32 j);
  BaseFloat& Distance(int32 i, int32 j) {
    KALDI_ASSERT(i < npoints_ && j < i);
    return dist_vec_[(i * (i - 1)) / 2 + j];
  }

  BaseFloat ans_;
  const std::vector<Clusterable*> &points_;
  BaseFloat max_merge_thresh_;
  int32 min_clust_;
  std::vector<Clusterable*> *clusters_;
  std::vector<int32> *assignments_;
  size_t tree_index_;

  std::vector<Clusterable*> tmp_clusters_;
  std::vector<int32> tmp_assignments_;

  std::vector<BaseFloat> dist_vec_;
  int32 nclusters_;
  int32 npoints_;
  typedef
    std::pair<BaseFloat, std::pair<uint_smaller, uint_smaller> > QueueElement;
  // Priority queue using greater (lowest distances are highest priority).
  typedef std::priority_queue<QueueElement, std::vector<QueueElement>,
      std::greater<QueueElement>  > QueueType;
  QueueType queue_;
};

BaseFloat BottomUpClustererEntropy::Cluster() {
  KALDI_VLOG(2) << "Initializing cluster assignments.";
  InitializeAssignments();
  KALDI_VLOG(2) << "Setting initial distances.";
  SetInitialDistances();

  KALDI_VLOG(2) << "Clustering...";
  while (nclusters_ > min_clust_ && !queue_.empty()) {
    std::pair<BaseFloat, std::pair<uint_smaller, uint_smaller> > pr = queue_.top();
    BaseFloat dist = pr.first;
    int32 i = (int32) pr.second.first, j = (int32) pr.second.second;
    queue_.pop();
    if (CanMerge(i, j, dist)) MergeClusters(i, j);
  }
  KALDI_VLOG(2) << "Renumbering clusters to contiguous numbers.";
  Renumber();
  return ans_;
}

void BottomUpClustererEntropy::Renumber() {
  KALDI_VLOG(2) << "Freeing up distance vector.";
  {
    vector<BaseFloat> tmp;
    tmp.swap(dist_vec_);
  }

  // called after clustering, renumbers to make clusters contiguously
  // numbered. also processes assignments_ to remove chains of references.
  KALDI_VLOG(2) << "Creating new copy of non-NULL clusters.";

  // mapping from intermediate to final clusters.
  std::vector<uint_smaller> mapping(npoints_, static_cast<uint_smaller> (-1)); 
  std::vector<Clusterable*> new_clusters(nclusters_);
  int32 clust = 0;
  for (int32 i = 0; i < npoints_; i++) {
    if ((*clusters_)[i] != NULL) {
      KALDI_ASSERT(clust < nclusters_);
      new_clusters[clust] = (*clusters_)[i];
      mapping[i] = clust;
      clust++;
    }
  }
  KALDI_ASSERT(clust == nclusters_);

  KALDI_VLOG(2) << "Creating new copy of assignments.";
  std::vector<int32> new_assignments(npoints_);
  for (int32 i = 0; i < npoints_; i++) {  // now reprocess assignments_.
    int32 ii = i;
    while ((*assignments_)[ii] != ii)
      ii = (*assignments_)[ii];  // follow the chain.
    KALDI_ASSERT((*clusters_)[ii] != NULL);  // cannot have assignment to nonexistent cluster.
    KALDI_ASSERT(mapping[ii] != static_cast<uint_smaller>(-1));
    new_assignments[i] = mapping[ii];
  }
  clusters_->swap(new_clusters);
  assignments_->swap(new_assignments);
}

void BottomUpClustererEntropy::InitializeAssignments() {
  clusters_->resize(npoints_);
  assignments_->resize(npoints_);
  for (int32 i = 0; i < npoints_; i++) {  // initialize as 1-1 mapping.
    (*clusters_)[i] = points_[i]->Copy();
    (*assignments_)[i] = i;
  }
}

void BottomUpClustererEntropy::SetInitialDistances() {
  for (int32 i = 0; i < npoints_; i++) {
    for (int32 j = 0; j < i; j++) {
      BaseFloat dist = 
        dynamic_cast<EntropyClusterable*>((*clusters_)[i])->DistanceEntropy(
                                    *((*clusters_)[j]), tree_index_);
      dist_vec_[(i * (i - 1)) / 2 + j] = dist;
      if (dist <= max_merge_thresh_)
        queue_.push(std::make_pair(dist, 
                                   std::make_pair(static_cast<uint_smaller>(i),
            static_cast<uint_smaller>(j))));
    }
  }
}

bool BottomUpClustererEntropy::CanMerge(int32 i, int32 j, BaseFloat dist) {
  KALDI_ASSERT(i != j && i < npoints_ && j < npoints_);
  if ((*clusters_)[i] == NULL || (*clusters_)[j] == NULL)
    return false;
  BaseFloat cached_dist = dist_vec_[(i * (i - 1)) / 2 + j];
  return (std::fabs(cached_dist - dist) <= 1.0e-05 * std::fabs(dist));
}

void BottomUpClustererEntropy::MergeClusters(int32 i, int32 j) {
  KALDI_ASSERT(i != j && i < npoints_ && j < npoints_);
  dynamic_cast<EntropyClusterable*>(
                    (*clusters_)[i])->AddAndMerge(*((*clusters_)[j]),
                                                  tree_index_);
  delete (*clusters_)[j];
  (*clusters_)[j] = NULL;
  // note that we may have to follow the chain within "assignment_" to get
  // final assignments.
  (*assignments_)[j] = i;
  // subtract negated objective function change, i.e. add objective function
  // change.
  ans_ -= dist_vec_[(i * (i - 1)) / 2 + j];
  nclusters_--;
  // Now update "distances".
  for (int32 k = 0; k < npoints_; k++) {
    if (k != i && (*clusters_)[k] != NULL) {
      if (k < i)
        SetDistance(i, k);  // SetDistance requires k < i.
      else
        SetDistance(k, i);
    }
  }
}

void BottomUpClustererEntropy::ReconstructQueue() {
  // empty queue [since there is no clear()]
  {
    QueueType tmp;
    std::swap(tmp, queue_);
  }
  for (int32 i = 0; i < npoints_; i++) {
    if ((*clusters_)[i] != NULL) {
      for (int32 j = 0; j < i; j++) {
        if ((*clusters_)[j] != NULL) {
          BaseFloat dist = dist_vec_[(i * (i - 1)) / 2 + j];
          if (dist <= max_merge_thresh_) {
            queue_.push(std::make_pair(dist, std::make_pair(
                static_cast<uint_smaller>(i), static_cast<uint_smaller>(j))));
          }
        }
      }
    }
  }
}

void BottomUpClustererEntropy::SetDistance(int32 i, int32 j) {
  KALDI_ASSERT(i < npoints_ && j < i && (*clusters_)[i] != NULL
         && (*clusters_)[j] != NULL);
  BaseFloat dist = 
    dynamic_cast<EntropyClusterable*>((*clusters_)[i])
    ->DistanceEntropy(*((*clusters_)[j]), tree_index_);
  dist_vec_[(i * (i - 1)) / 2 + j] = dist;  // set the distance in the array.
  if (dist < max_merge_thresh_) {
    queue_.push(std::make_pair(dist, std::make_pair(static_cast<uint_smaller>(i),
        static_cast<uint_smaller>(j))));
  }
  // every time it's at least twice the maximum possible size.
  if (queue_.size() >= static_cast<size_t> (npoints_ * npoints_)) {
    // Control memory use by getting rid of orphaned queue entries
    ReconstructQueue();
  }
}


BaseFloat ClusterBottomUpEntropy(const std::vector<Clusterable*> &points,
                          BaseFloat max_merge_thresh,
                          int32 min_clust,
                          std::vector<Clusterable*> *clusters_out,
                          std::vector<int32> *assignments_out,
                          size_t tree_index) {
  KALDI_ASSERT(max_merge_thresh >= 0.0 && min_clust >= 0);
  KALDI_ASSERT(!ContainsNullPointers(points));
  int32 npoints = points.size();
  // make sure fits in uint_smaller and does not hit the -1 which is reserved.
  KALDI_ASSERT(sizeof(uint_smaller)==sizeof(uint32) ||
               npoints < static_cast<int32>(static_cast<uint_smaller>(-1)));

  KALDI_VLOG(2) << "Initializing clustering object.";
  BottomUpClustererEntropy bc(points, max_merge_thresh,
                              min_clust, clusters_out,
                              assignments_out, tree_index);
  BaseFloat ans = bc.Cluster();
  if (clusters_out) KALDI_ASSERT(!ContainsNullPointers(*clusters_out));
  return ans;
}

} // end of namespace kaldi
