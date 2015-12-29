#include "tree/build-tree-expand.h"
#include "tree/event-map.h"

namespace kaldi {

vector<EventMap*> ExpandDecisionTree(const ContextDependency &ctx_dep,
                                     const BuildTreeStatsType &stats,
                                     const Questions &qo,
                                     int32 num_qst) {
  vector<EventMap*> ans(num_qst, NULL);

  int32 N = ctx_dep.ContextWidth();

  for (int i = 0; i < num_qst; i++) {
    ans[i] = ctx_dep.ToPdfMap().Copy();
  }

  vector<BuildTreeStatsType> splits;
  SplitStatsByMap(stats, ctx_dep.ToPdfMap(), &splits);

  int num_leaves = ctx_dep.NumPdfs();
  KALDI_ASSERT(num_leaves == splits.size());

  vector<QuestionsForKey> qs;
  for (int n = 0; n < N; n++) {
    QuestionsForKey qk = qo.GetQuestionsOf(n);
    qs.push_back(qk);
  }

  for (int i = 0; i < num_leaves; i++) {
  }

  return ans;
}

}  // namespace kaldi
