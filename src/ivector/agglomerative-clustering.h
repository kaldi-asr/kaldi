// ivector/agglomerative-clustering.h

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
      BaseFloat thresh,
      int32 min_clust,
      std::vector<int32> *assignments_out)
      : count_(0), costs_(costs), thresh_(thresh), min_clust_(min_clust),
        assignments_(assignments_out) {
    num_clusters_ = costs.NumRows();
    num_points_ = costs.NumRows();
  }

  // Performs the clustering
  void Cluster();
 private:
  // Returns the cost between clusters with IDs i and j
  BaseFloat GetCost(int32 i, int32 j);
  // Initializes the clustering queue with singleton clusters
  void Initialize();
  // Merges clusters with IDs i and j and updates cost map and queue
  void MergeClusters(int32 i, int32 j);


  int32 count_;  // Count of clusters that have been created. Also used to give
                 // clusters unique IDs.
  const Matrix<BaseFloat> &costs_;  // cost matrix
  BaseFloat thresh_;  // stopping criterion threshold
  int32 min_clust_;  // minimum number of clusters
  std::vector<int32> *assignments_;  // assignments out

  // Priority queue using greater (lowest costs are highest priority).
  // Elements contain pairs of cluster IDs and their cost.
  typedef std::pair<BaseFloat, std::pair<uint16,
    uint16> > QueueElement;
  typedef std::priority_queue<QueueElement, std::vector<QueueElement>,
    std::greater<QueueElement>  > QueueType;
  QueueType queue_;

  // Map from cluster IDs to cost between them
  std::unordered_map<std::pair<int32, int32>, BaseFloat,
                     PairHasher<int32, int32>> cluster_cost_map_;
  // Map from cluster ID to cluster object address
  std::unordered_map<int32, AhcCluster*> clusters_map_;
  std::set<int32> active_clusters_;  // IDs of unmerged clusters
  int32 num_clusters_;  // number of active clusters
  int32 num_points_;  // total number of points to cluster
};

/** This is the function that is called to perform the agglomerative
 *  clustering. It takes the following arguments:
 *   - A matrix of all pairwise costs, with each row/column corresponding
 *      to an utterance ID, and the elements of the matrix containing the
        cost for pairing the utterances for its row and column
 *   - A threshold which is used as the stopping criterion for the clusters
 *   - A minimum number of clusters that will not be merged past
 *   - A vector which will be filled with integer IDs corresponding to each
 *      of the rows/columns of the score matrix.
 *
 *  The basic algorithm is as follows:
 *  \code
 *      while (num-clusters > min_clust && smallest-merge-cost <= thresh)
 *          merge the two clusters with lowest cost.
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
 */
void AgglomerativeCluster(
    const Matrix<BaseFloat> &costs,
    BaseFloat thresh,
    int32 min_clust,
    std::vector<int32> *assignments_out);

}  // end namespace kaldi.

#endif  // KALDI_IVECTOR_AGGLOMERATIVE_CLUSTERING_H_
