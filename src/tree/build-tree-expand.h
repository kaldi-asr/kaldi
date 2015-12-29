// tree/build-tree-virtual.h

// Hainan Xu

#ifndef KALDI_TREE_BUILD_TREE_EXPAND_H_
#define KALDI_TREE_BUILD_TREE_EXPAND_H_

#include "tree/build-tree-utils.h"
#include "tree/context-dep.h"
#include "util/stl-utils.h"
#include <queue>

using std::vector;
using std::set;
using std::map;
using std::priority_queue;
using std::string;

namespace kaldi {

vector<EventMap*> ExpandDecisionTree(const ContextDependency &ctx_dep,
                                     const BuildTreeStatsType &stats,
                                     const Questions &qo,
                                     int32 num_qst);

}  // namespace kaldi

#endif

