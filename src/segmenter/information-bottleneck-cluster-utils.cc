// segmenter/information-bottleneck-cluster-utils.cc

// Copyright 2017   Vimal Manohar (Johns Hopkins University)

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

#include "base/kaldi-common.h"
#include "tree/cluster-utils.h"
#include "segmenter/information-bottleneck-cluster-utils.h"

namespace kaldi {
  
typedef uint16 uint_smaller;
typedef int16 int_smaller;

class InformationBottleneckBottomUpClusterer : public BottomUpClusterer {
 public:
  InformationBottleneckBottomUpClusterer(
      const std::vector<Clusterable*> &points,
      const InformationBottleneckClustererOptions &opts,
      BaseFloat max_merge_thresh,
      int32 min_clusters,
      std::vector<Clusterable*> *clusters_out,
      std::vector<int32> *assignments_out);
  
 private:
  virtual BaseFloat ComputeDistance(int32 i, int32 j);
  virtual bool StoppingCriterion() const;
  virtual void UpdateClustererStats(int32 i, int32 j);

  BaseFloat NormalizedMutualInformation() const {
    return ((merged_entropy_ - current_entropy_) 
            / (merged_entropy_ - initial_entropy_));
  }

  /// Stop merging when the stopping criterion, e.g. NMI, reaches this 
  /// threshold.
  BaseFloat stopping_threshold_;

  /// Weight of the relevant variables entropy towards the objective.
  BaseFloat relevance_factor_;
  
  /// Weight of the input variables entropy towards the objective.
  BaseFloat input_factor_;

  /// Running entropy of the clusters.
  BaseFloat current_entropy_;

  /// Some stats computed by the constructor that will be useful for 
  /// adding stopping criterion.
  BaseFloat initial_entropy_;
  BaseFloat merged_entropy_;
};


InformationBottleneckBottomUpClusterer::InformationBottleneckBottomUpClusterer(
    const std::vector<Clusterable*> &points,
    const InformationBottleneckClustererOptions &opts,
    BaseFloat max_merge_thresh,
    int32 min_clusters,
    std::vector<Clusterable*> *clusters_out,
    std::vector<int32> *assignments_out) :
      BottomUpClusterer(points, max_merge_thresh, min_clusters,
                        clusters_out, assignments_out),
      stopping_threshold_(opts.stopping_threshold), 
      relevance_factor_(opts.relevance_factor), 
      input_factor_(opts.input_factor),
      current_entropy_(0.0), initial_entropy_(0.0), merged_entropy_(0.0) {
  if (points.size() == 0) return;
  
  InformationBottleneckClusterable* ibc = 
    static_cast<InformationBottleneckClusterable*>(points[0]->Copy());
  initial_entropy_ -= ibc->Objf(1.0, 0.0);

  for (size_t i = 1; i < points.size(); i++) {
    InformationBottleneckClusterable *c =
      static_cast<InformationBottleneckClusterable*>(points[i]);
    ibc->Add(*points[i]);
    initial_entropy_ -= c->Objf(1.0, 0.0);
  }

  merged_entropy_ = -ibc->Objf(1.0, 0.0);
  current_entropy_ = initial_entropy_;
}

BaseFloat InformationBottleneckBottomUpClusterer::ComputeDistance(
    int32 i, int32 j) {
  const InformationBottleneckClusterable* cluster_i
    = static_cast<const InformationBottleneckClusterable*>(GetCluster(i));
  const InformationBottleneckClusterable* cluster_j
    = static_cast<const InformationBottleneckClusterable*>(GetCluster(j));

  BaseFloat dist = (cluster_i->Distance(*cluster_j, relevance_factor_, 
                                        input_factor_));
                    // / (cluster_i->Normalizer() + cluster_j->Normalizer()));
  Distance(i, j) = dist;  // set the distance in the array.
  return dist;
}

bool InformationBottleneckBottomUpClusterer::StoppingCriterion() const { 
  bool flag = (NumClusters() <= MinClusters() || IsQueueEmpty() || 
               NormalizedMutualInformation() < stopping_threshold_);
  if (GetVerboseLevel() < 2 || !flag) return flag;

  if (NormalizedMutualInformation() < stopping_threshold_) {
    KALDI_VLOG(2) << "Stopping at " << NumClusters() << " clusters "
                  << "because NMI = " << NormalizedMutualInformation()
                  << " < stopping_threshold (" << stopping_threshold_ << ")";
  } else if (NumClusters() < MinClusters()) {
    KALDI_VLOG(2) << "Stopping at " << NumClusters() << " clusters "
                  << "<= min-clusters (" << MinClusters() << ")";
  } else if (IsQueueEmpty()) {
    KALDI_VLOG(2) << "Stopping at " << NumClusters() << " clusters "
                  << "because queue is empty.";
  }

  return flag;
}

void InformationBottleneckBottomUpClusterer::UpdateClustererStats(
    int32 i, int32 j) {
  const InformationBottleneckClusterable* cluster_i
    = static_cast<const InformationBottleneckClusterable*>(GetCluster(i));
  current_entropy_ += cluster_i->Distance(*GetCluster(j), 1.0, 0.0);

  if (GetVerboseLevel() > 2) {
    const InformationBottleneckClusterable* cluster_j
      = static_cast<const InformationBottleneckClusterable*>(GetCluster(j));
    std::vector<int32> cluster_i_points;
    {
      std::map<int32, BaseFloat>::const_iterator it 
        = cluster_i->Counts().begin();
      for (; it != cluster_i->Counts().end(); ++it) 
        cluster_i_points.push_back(it->first);
    }

    std::vector<int32> cluster_j_points;
    {
      std::map<int32, BaseFloat>::const_iterator it 
        = cluster_j->Counts().begin();
      for (; it != cluster_j->Counts().end(); ++it) 
        cluster_j_points.push_back(it->first);
    }
    KALDI_VLOG(3) << "Merging clusters " 
                  << "(" << cluster_i_points
                  << ", " << cluster_j_points
                  << ").. distance=" << Distance(i, j)
                  << ", num-clusters-after-merge= " << NumClusters() - 1
                  << ", NMI= " << NormalizedMutualInformation();
  }
}

BaseFloat IBClusterBottomUp(
    const std::vector<Clusterable*> &points,
    const InformationBottleneckClustererOptions &opts,
    BaseFloat max_merge_thresh,
    int32 min_clust,
    std::vector<Clusterable*> *clusters_out,
    std::vector<int32> *assignments_out) {
  KALDI_ASSERT(max_merge_thresh >= 0.0 && min_clust >= 0);
  KALDI_ASSERT(opts.stopping_threshold >= 0.0);
  KALDI_ASSERT(opts.relevance_factor >= 0.0 && opts.input_factor >= 0.0);

  KALDI_ASSERT(!ContainsNullPointers(points));
  int32 npoints = points.size();
  // make sure fits in uint_smaller and does not hit the -1 which is reserved.
  KALDI_ASSERT(sizeof(uint_smaller)==sizeof(uint32) ||
               npoints < static_cast<int32>(static_cast<uint_smaller>(-1)));

  KALDI_VLOG(2) << "Initializing clustering object.";
  InformationBottleneckBottomUpClusterer bc(
      points, opts, max_merge_thresh, min_clust, 
      clusters_out, assignments_out);
  BaseFloat ans = bc.Cluster();
  if (clusters_out) KALDI_ASSERT(!ContainsNullPointers(*clusters_out));
  return ans;
}

}  // end namespace kaldi
