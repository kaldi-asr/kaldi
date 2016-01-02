#include "tree/build-tree-expand.h"
#include "tree/event-map.h"

namespace kaldi {

vector<EventMap*> ExpandDecisionTree(const ContextDependency &ctx_dep,
                                     const BuildTreeStatsType &stats,
                                     const Questions &qo,
                                     int32 num_qst) {
  vector<EventMap*> ans(num_qst, NULL);

  // int32 N = ctx_dep.ContextWidth();

  for (int i = 0; i < num_qst; i++) {
    ans[i] = ctx_dep.ToPdfMap().Copy();
  }

  vector<BuildTreeStatsType> splits;
  SplitStatsByMap(stats, ctx_dep.ToPdfMap(), &splits);

  int num_leaves = ctx_dep.NumPdfs();
  KALDI_ASSERT(num_leaves == splits.size());

  for (int l = 0; l < num_leaves; l++) {
    // process stats mapped to the l'th leaf, i.e. in splits[l]
    std::vector<EventKeyType> all_keys;
    qo.GetKeysWithQuestions(&all_keys);

    if (all_keys.size() == 0) {
      KALDI_WARN << "ExpandDecisionTree(), no keys available to split "
       " on (maybe no key covered all of your events, or there was a problem"
       " with your questions configuration?)";
    }

    vector<vector<EventValueType> > temp_yes_set_vec;
    vector<BaseFloat> improvement_vec;
    vector<EventKeyType> keys;

    for (size_t i = 0; i < all_keys.size(); i++) {
      if (qo.HasQuestionsForKey(all_keys[i])) {
        FindNBestSplitsForKey(num_qst, splits[l], qo,
                              all_keys[i], &temp_yes_set_vec, &improvement_vec);
        keys.resize(improvement_vec.size(), all_keys[i]);
      }
    }
    // TODO(hxu) now there are more questions than we want, might be better
    // to pick the top questions (or not..)

    for (int i = 0; i < 
  }

  return ans;
}

}  // namespace kaldi
