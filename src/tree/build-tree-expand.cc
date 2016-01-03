#include "tree/event-map.h"
#include "tree/build-tree-utils.h"
#include "tree/build-tree-expand.h"

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

  std::vector<EventKeyType> all_keys;
  qo.GetKeysWithQuestions(&all_keys);

  vector<vector<key_yesset> > questions_for_trees(num_qst * all_keys.size());

  for (int l = 0; l < num_leaves; l++) {
    // process stats mapped to the l'th leaf, i.e. in splits[l]

    if (all_keys.size() == 0) {
      KALDI_WARN << "ExpandDecisionTree(), no keys available to split "
       " on (maybe no key covered all of your events, or there was a problem"
       " with your questions configuration?)";
    }

    vector<vector<EventValueType> > temp_yes_set_vec;
    vector<BaseFloat> improvement_vec;
//    vector<EventKeyType> keys;

    for (size_t i = 0; i < all_keys.size(); i++) {
      if (qo.HasQuestionsForKey(all_keys[i])) {
        FindNBestSplitsForKey(num_qst, splits[l], qo,
                              all_keys[i], &temp_yes_set_vec, &improvement_vec);

        // if vector is smaller than expected size, fill it with the
        // 1st question
      }
      temp_yes_set_vec.resize((i + 1) * num_qst, temp_yes_set_vec[i * num_qst]);
    }
    // here the size of temp_yes_set_vec is all_keys.size() * num_qst
    // and for each key there are num_qst questions
    KALDI_ASSERT(temp_yes_set_vec.size() == all_keys.size() * num_qst);

    for (int j = 0; j < num_qst * all_keys.size(); j++) {
      key_yesset ky;
      ky.key = all_keys[j / num_qst];
      ky.yes_set = temp_yes_set_vec[j];

      // the index for the newly added entry would be l (the outer-most loop)
      questions_for_trees[j].push_back(ky);
    }
  }

  for (int i = 0; i < num_leaves; i++) {
    EventAnswerType next = num_leaves;
    KALDI_ASSERT(questions_for_trees[i].size() == num_leaves);
    ans[i]->ExpandTree(questions_for_trees[i], &next);
  }
  return ans;
}

}  // namespace kaldi
