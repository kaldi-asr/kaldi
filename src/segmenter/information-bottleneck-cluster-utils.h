// segmenter/information-bottleneck-cluster-utils.h

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

#ifndef KALDI_SEGMENTER_INFORMATION_BOTTLENECK_CLUSTER_UTILS_H_
#define KALDI_SEGMENTER_INFORMATION_BOTTLENECK_CLUSTER_UTILS_H_

#include "base/kaldi-common.h"
#include "tree/cluster-utils.h"
#include "segmenter/information-bottleneck-clusterable.h"
#include "util/common-utils.h"

namespace kaldi {

struct InformationBottleneckClustererOptions {
  BaseFloat distance_threshold;
  int32 num_clusters;
  BaseFloat stopping_threshold;
  BaseFloat relevance_factor;
  BaseFloat input_factor;
  bool normalize_by_count;
  bool normalize_by_entropy;

  InformationBottleneckClustererOptions() :
    distance_threshold(std::numeric_limits<BaseFloat>::max()), num_clusters(1),
    stopping_threshold(0.3), relevance_factor(1.0), input_factor(0.1),
    normalize_by_count(false), normalize_by_entropy(false) { }


  void Register(OptionsItf *opts) {
    opts->Register("stopping-threshold", &stopping_threshold,
                   "Stopping merging/splitting when an objective such as "
                   "NMI reaches this value.");
    opts->Register("relevance-factor", &relevance_factor,
                   "Weight factor of the entropy of relevant variables "
                   "in the objective function");
    opts->Register("input-factor", &input_factor,
                   "Weight factor of the entropy of input variables "
                   "in the objective function");
    opts->Register("normalize-by-count", &normalize_by_count,
                   "If provided, normalizes the score (distance) by "
                   "the count post-merge.");
    opts->Register("normalize-by-entropy", &normalize_by_entropy,
                   "If provided, normalizes the score (distance) by "
                   "the entropy post-merge.");
  }
};

BaseFloat IBClusterBottomUp(
    const std::vector<Clusterable*> &points,
    const InformationBottleneckClustererOptions &opts,
    BaseFloat max_merge_thresh,
    int32 min_clusters,
    std::vector<Clusterable*> *clusters_out,
    std::vector<int32> *assignments_out);

}  // end namespace kaldi

#endif  // KALDI_SEGMENTER_INFORMATION_BOTTLENECK_CLUSTER_UTILS_H_
