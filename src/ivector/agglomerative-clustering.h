// ivector/agglomerative-clustering.h

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

#ifndef KALDI_IVECTOR_AGGLOMERATIVE_CLUSTERING_H_
#define KALDI_IVECTOR_AGGLOMERATIVE_CLUSTERING_H_

#include <vector>
#include <queue>
#include <set>
#include <unordered_map>
#include <functional>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "util/stl-utils.h"

namespace kaldi {

/// AhcCluster is the cluster object for the agglomerative clustering. It
/// contains three integer IDs: its own ID and the IDs of its "parents", i.e.
/// the clusters that were merged to form it. It also contains the size (the
/// number of points in the cluster) and a vector of the IDs of the utterances
/// contained in the cluster.
struct AhcCluster {
  int32 id,
    parent1,
    parent2,
    size;
  std::vector<int32> utt_ids;
  AhcCluster(int32 id, int32 p1, int32 p2, std::vector<int32> utts)
      : id(id), parent1(p1), parent2(p2), utt_ids(utts) {
    size = utts.size();
  }
};

/// The AgglomerativeClusterer class contains the necessary mechanisms for the
/// actual clustering algorithm.
class AgglomerativeClusterer {
 public:
  AgglomerativeClusterer(
      const Matrix<BaseFloat> &costs,
      BaseFloat threshold,
      int32 min_clusters,
      int32 first_pass_max_points,
      BaseFloat max_cluster_fraction,
      std::vector<int32> *assignments_out)
      : costs_(costs), threshold_(threshold), min_clusters_(min_clusters),
        first_pass_max_points_(first_pass_max_points),
        assignments_(assignments_out) {
    num_points_ = costs.NumRows();

    // The max_cluster_size_ is a hard limit on the number points in a cluster.
    // This is useful for handling degenerate cases where some outlier points
    // form their own clusters and force everything else to be clustered
    // together, e.g. when min-clusters is provided instead of a threshold.
    max_cluster_size_ = ceil(num_points_ * max_cluster_fraction);

    // The count_, which is used for identifying clusters, is initialized to
    // num_points_ because cluster IDs 1..num_points_ are reserved for input
    // points, which are the initial set of clusters.
    count_ = num_points_;

    // The second_pass_count_, which is used for identifying the initial set of
    // second pass clusters and initializing count_ before the second pass, is
    // initialized to 0 and incremented whenever a new cluster is added to the
    // initial set of second pass clusters.
    second_pass_count_ = 0;
  }

  // Clusters points. Chooses single pass or two pass algorithm.
  void Cluster();

  // Clusters points using single pass algorithm.
  void ClusterSinglePass();

  // Clusters points using two pass algorithm.
  void ClusterTwoPass();

 private:
  // Encodes cluster pair into a 32bit unsigned integer.
  uint32 EncodePair(int32 i, int32 j);
  // Decodes cluster pair from a 32bit unsigned integer.
  std::pair<int32, int32> DecodePair(uint32 key);
  // Initializes the clustering queue with singleton clusters
  void InitializeClusters(int32 first, int32 last);
  // Does hierarchical agglomerative clustering
  void ComputeClusters(int32 min_clusters);
  // Adds clusters created in first pass to second pass clusters
  void AddClustersToSecondPass();
  // Assigns points to clusters
  void AssignClusters();
  // Merges clusters with IDs i and j and updates cost map and queue
  void MergeClusters(int32 i, int32 j);

  const Matrix<BaseFloat> &costs_;  // cost matrix
  BaseFloat threshold_;  // stopping criterion threshold
  int32 min_clusters_;  // minimum number of clusters
  int32 first_pass_max_points_;  // maximum number of points in each subset
  std::vector<int32> *assignments_;  // assignments out

  int32 num_points_;  // total number of points to cluster
  int32 max_cluster_size_;  // maximum number of points in a cluster
  int32 count_;  // count of first pass clusters, used for identifying clusters
  int32 second_pass_count_;  // count of second pass clusters

  // Priority queue using greater (lowest costs are highest priority).
  // Elements contain pairs of cluster IDs and their cost.
  typedef std::pair<BaseFloat, uint32> QueueElement;
  typedef std::priority_queue<QueueElement, std::vector<QueueElement>,
    std::greater<QueueElement>  > QueueType;
  QueueType queue_, second_pass_queue_;

  // Map from cluster IDs to cost between them
  std::unordered_map<uint32, BaseFloat> cluster_cost_map_;
  // Map from cluster ID to cluster object address
  std::unordered_map<int32, AhcCluster*> clusters_map_;
  // Set of unmerged cluster IDs
  std::set<int32> active_clusters_;

  // Map from second pass cluster IDs to cost between them
  std::unordered_map<uint32, BaseFloat> second_pass_cluster_cost_map_;
  // Map from second pass cluster ID to cluster object address
  std::unordered_map<int32, AhcCluster*> second_pass_clusters_map_;
  // Set of unmerged second pass cluster IDs
  std::set<int32> second_pass_active_clusters_;
};

/** This is the function that is called to perform the agglomerative
 *  clustering. It takes the following arguments:
 *   - A matrix of all pairwise costs, with each row/column corresponding
 *      to an utterance ID, and the elements of the matrix containing the
 *      cost for pairing the utterances for its row and column
 *   - A threshold which is used as the stopping criterion for the clusters
 *   - A minimum number of clusters that will not be merged past
 *   - A maximum fraction of points that can be in a cluster
 *   - A vector which will be filled with integer IDs corresponding to each
 *      of the rows/columns of the score matrix.
 *
 *  The basic algorithm is as follows:
 *  \code
 *      while (num-clusters > min-clusters && smallest-merge-cost <= threshold)
 *          if (size-of-new-cluster <= max-cluster-size)
 *              merge the two clusters with lowest cost
 *  \endcode
 *
 *  The cost between two clusters is the average cost of all pairwise
 *  costs between points across the two clusters.
 *
 *  The algorithm takes advantage of the fact that the sum of the pairwise
 *  costs between the points of clusters I and J is equiavlent to the
 *  sum of the pairwise costs between cluster I and the parents of cluster
 *  J. In other words, the total cost between I and J is the sum of the
 *  costs between clusters I and M and clusters I and N, where
 *  cluster J was formed by merging clusters M and N.
 *
 *  If the number of points to cluster is larger than first-pass-max-points,
 *  then clustering is done in two passes. In the first pass, input points are
 *  divided into contiguous subsets of size at most first-pass-max-points and
 *  each subset is clustered separately. In the second pass, the first pass
 *  clusters are merged into the final set of clusters.
 *
 */
void AgglomerativeCluster(
    const Matrix<BaseFloat> &costs,
    BaseFloat threshold,
    int32 min_clusters,
    int32 first_pass_max_points,
    BaseFloat max_cluster_fraction,
    std::vector<int32> *assignments_out);

}  // end namespace kaldi.

#endif  // KALDI_IVECTOR_AGGLOMERATIVE_CLUSTERING_H_
