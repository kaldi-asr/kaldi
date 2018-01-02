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
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "util/stl-utils.h"

namespace kaldi {

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

class AgglomerativeClusterer {
 public:
  AgglomerativeClusterer(
      const Matrix<BaseFloat> &scores,
      BaseFloat thresh,
      int32 min_clust,
      std::vector<int32> *assignments_out)
      : ct_(0), scores_(scores), thresh_(thresh), min_clust_(min_clust),
        assignments_(assignments_out) {
    nclusters_ = scores.NumRows();
    npoints_ = scores.NumRows();
  }
  void Cluster();
 private:
  BaseFloat ScoreLookup(int32 i, int32 j);
  void Initialize();
  void MergeClusters(int32 i, int32 j);


  int32 ct_;
  const Matrix<BaseFloat> &scores_;
  BaseFloat thresh_;
  int32 min_clust_;
  std::vector<int32> *assignments_;
  typedef std::pair<BaseFloat, std::pair<uint16,
    uint16> > QueueElement;
  typedef std::priority_queue<QueueElement, std::vector<QueueElement>,
    std::greater<QueueElement>  > QueueType;
  std::unordered_map<std::pair<int32, int32>, BaseFloat,
                     PairHasher<int32, int32>> cluster_score_map_;
  std::unordered_map<int32, AhcCluster*> clusters_map_;
  std::set<int32> active_clusters_;
  int32 nclusters_;
  int32 npoints_;
  // Priority queue using greater (lowest distances are highest priority).
  QueueType queue_;
};

void AgglomerativeCluster(
    const Matrix<BaseFloat> &scores,
    BaseFloat thresh,
    int32 min_clust,
    std::vector<int32> *assignments_out);

}  // end namespace kaldi.

#endif  // KALDI_IVECTOR_AGGLOMERATIVE_CLUSTERING_H_
