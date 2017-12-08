// ivector/agglomerative-bottom-up-clustering.h

// Copyright 2016  David Snyder
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

#ifndef KALDI_IVECTOR_AGGLOMERATIVE_BOTTOM_UP_CLUSTERING_H_
#define KALDI_IVECTOR_AGGLOMERATIVE_BOTTOM_UP_CLUSTERING_H_

#include <vector>
#include <set>
#include <unordered_map>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "util/stl-utils.h"

namespace kaldi {

void AgglomerativeClusterBottomUp(
    const std::vector<std::string> &uttlist,
    const std::unordered_map<std::string, BaseFloat> &score_map,
    BaseFloat max_dist,
    BaseFloat thresh,
    int32 min_clust,
    std::vector<int32> *assignments_out);

}  // end namespace kaldi.

#endif  // KALDI_IVECTOR_AGGLOMERATIVE_BOTTOM_UP_CLUSTERING_H_
