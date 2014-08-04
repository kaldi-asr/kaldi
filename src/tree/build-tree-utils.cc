// tree/build-tree-utils.cc

// Copyright 2009-2011  Microsoft Corporation;  Haihua Xu

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

#include <set>
#include <queue>
#include "util/stl-utils.h"
#include "tree/build-tree-utils.h"



namespace kaldi {

void WriteBuildTreeStats(std::ostream &os, bool binary, const BuildTreeStatsType &stats) {
  WriteToken(os, binary, "BTS");
  uint32 size = stats.size();
  WriteBasicType(os, binary, size);
  for (size_t i = 0; i < size; i++) {
    WriteEventType(os, binary, stats[i].first);
    bool nonNull = (stats[i].second != NULL);
    WriteBasicType(os, binary, nonNull);
    if (nonNull) stats[i].second->Write(os, binary);
  }
  if (os.fail()) {
    KALDI_ERR << "WriteBuildTreeStats: write failed.";
  }
  if (!binary) os << '\n';
}


void ReadBuildTreeStats(std::istream &is, bool binary, const Clusterable &example, BuildTreeStatsType *stats) {
  KALDI_ASSERT(stats != NULL);
  KALDI_ASSERT(stats->empty());
  ExpectToken(is, binary, "BTS");
  uint32 size;
  ReadBasicType(is, binary, &size);
  stats->resize(size);
  for (size_t i = 0; i < size; i++) {
    ReadEventType(is, binary, &((*stats)[i].first));
    bool nonNull;
    ReadBasicType(is, binary, &nonNull);
    if (nonNull) (*stats)[i].second = example.ReadNew(is, binary);
    else (*stats)[i].second = NULL;
  }
}


bool PossibleValues(EventKeyType key,
                    const BuildTreeStatsType &stats,
                    std::vector<EventValueType> *ans) {
  bool all_present = true;
  std::set<EventValueType> values;
  BuildTreeStatsType::const_iterator iter = stats.begin(), end = stats.end();
  for (; iter != end; ++iter) {
    EventValueType val;
    if (EventMap::Lookup(iter->first, key, &val))
      values.insert(val);
    else
      all_present = false;
  }
  if (ans)
    CopySetToVector(values, ans);
  return all_present;
}

static void GetEventKeys(const EventType &vec, std::vector<EventKeyType> *keys) {
  keys->resize(vec.size());
  EventType::const_iterator iter = vec.begin(), end = vec.end();
  std::vector<EventKeyType>::iterator out_iter = keys->begin();
  for (; iter!= end; ++iter, ++out_iter)
    *out_iter = iter->first;
}


// recall:
// typedef std::vector<std::pair<EventType, Clusterable*> > BuildTreeStatsType;
void FindAllKeys(const BuildTreeStatsType &stats, AllKeysType keys_type, std::vector<EventKeyType> *keys_out) {
  KALDI_ASSERT(keys_out != NULL);
  BuildTreeStatsType::const_iterator iter = stats.begin(), end = stats.end();
  if (iter == end) return;  // empty set of keys.
  std::vector<EventKeyType> keys;
  GetEventKeys(iter->first, &keys);
  ++iter;
  for (; iter!= end; ++iter) {
    std::vector<EventKeyType> keys2;
    GetEventKeys(iter->first, &keys2);
    if (keys_type == kAllKeysInsistIdentical) {
      if (keys2 != keys)
        KALDI_ERR << "AllKeys: keys in events are not all the same [called with kAllKeysInsistIdentical and all are not identical.";
    } else if (keys_type == kAllKeysIntersection) {
      std::vector<EventKeyType> new_keys(std::max(keys.size(), keys2.size()));
      // following line relies on sorted order of event keys.
      std::vector<EventKeyType>::iterator end_iter =
          std::set_intersection(keys.begin(), keys.end(), keys2.begin(), keys2.end(), new_keys.begin());
      new_keys.erase(end_iter, new_keys.end());
      keys = new_keys;
    } else {  // union.
      KALDI_ASSERT(keys_type == kAllKeysUnion);
      std::vector<EventKeyType> new_keys(keys.size()+keys2.size());
      // following line relies on sorted order of event keys.
      std::vector<EventKeyType>::iterator end_iter =
          std::set_union(keys.begin(), keys.end(), keys2.begin(), keys2.end(), new_keys.begin());
      new_keys.erase(end_iter, new_keys.end());
      keys = new_keys;
    }
  }
  *keys_out = keys;
}


EventMap *DoTableSplit(const EventMap &orig, EventKeyType key,  const BuildTreeStatsType &stats,
                       int32 *num_leaves) {
  // First-- map the stats to each leaf in the EventMap.
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, orig, &split_stats);
  // Now for each leaf that has stats in it, do the table split according to the given name.
  std::vector<EventMap*> splits(split_stats.size(), NULL);
  for (EventAnswerType leaf = 0; leaf < (EventAnswerType)split_stats.size(); leaf++) {
    if (!split_stats[leaf].empty()) {
      // first work out the possible values the name takes.
      std::vector<EventValueType> vals;  // vals are put here, sorted.
      bool all_present = PossibleValues(key, split_stats[leaf], &vals);
      KALDI_ASSERT(all_present);  // currently do not support mapping undefined values.
      KALDI_ASSERT(!vals.empty() && vals.front() >= 0);  // don't support mapping negative values
      // at present time-- would need different EventMap type, not TableEventMap.
      std::vector<EventMap*> table(vals.back()+1, (EventMap*)NULL);
      for (size_t idx = 0;idx < vals.size();idx++) {
        EventValueType val = vals[idx];
        if (idx == 0) table[val] = new ConstantEventMap(leaf);  // reuse current leaf.
        else table[val] = new ConstantEventMap( (*num_leaves)++ );  // else take new leaf id.
      }
      // takes ownershipof stats.
      splits[leaf] = new TableEventMap(key, table);
    }
  }
  EventMap *ans = orig.Copy(splits);
  DeletePointers(&splits);
  return ans;
}

EventMap *DoTableSplitMultiple(const EventMap &orig, const std::vector<EventKeyType> &keys,  const BuildTreeStatsType &stats, int32 *num_leaves) {

  if (keys.empty()) return orig.Copy();
  else {
    EventMap *cur = NULL;  // would make it &orig, except for const issues.
    for (size_t i = 0; i < keys.size(); i++) {
      EventMap *next = DoTableSplit( (cur ? *cur : orig), keys[i], stats, num_leaves);
      if (cur != NULL) delete cur;  // delete intermediate maps.
      cur = next;
    }
    return cur;
  }
}



void SplitStatsByMap(const BuildTreeStatsType &stats, const EventMap &e, std::vector<BuildTreeStatsType> *stats_out) {
  BuildTreeStatsType::const_iterator iter, end = stats.end();
  KALDI_ASSERT(stats_out != NULL);
  stats_out->clear();
  size_t size = 0;
  for (iter = stats.begin(); iter != end; ++iter) {
    const EventType &evec = iter->first;
    EventAnswerType ans;
    if (!e.Map(evec, &ans)) // this is an error--could not map it.
      KALDI_ERR << "SplitStatsByMap: could not map event vector " << EventTypeToString(evec)
                << "if error seen during tree-building, check that "
                << "--context-width and --central-position match stats, "
                << "and that phones that are context-independent (CI) during "
                << "stats accumulation do not share roots with non-CI phones.";
    size = std::max(size, (size_t)(ans+1));
  }
  stats_out->resize(size);
  for (iter = stats.begin(); iter != end; ++iter) {
    const EventType &evec = iter->first;
    EventAnswerType ans;
    bool b = e.Map(evec, &ans);
    KALDI_ASSERT(b);
    (*stats_out)[ans].push_back(*iter);
  }
}

void SplitStatsByKey(const BuildTreeStatsType &stats_in, EventKeyType key, std::vector<BuildTreeStatsType> *stats_out) {
  BuildTreeStatsType::const_iterator iter, end = stats_in.end();
  KALDI_ASSERT(stats_out != NULL);
  stats_out->clear();
  size_t size = 0;
  // This loop works out size of output vector.
  for (iter = stats_in.begin(); iter != end; ++iter) {
    const EventType &evec = iter->first;
    EventValueType val;
    if (! EventMap::Lookup(evec, key, &val)) // no such key.
      KALDI_ERR << "SplitStats: key "<< key << " is not present in event vector " << EventTypeToString(evec);
    size = std::max(size, (size_t)(val+1));
  }
  stats_out->resize(size);
  // This loop splits up the stats.
  for (iter = stats_in.begin(); iter != end; ++iter) {
    const EventType &evec = iter->first;
    EventValueType val;
    EventMap::Lookup(evec, key, &val);  // will not fail.
    (*stats_out)[val].push_back(*iter);
  }
}


void FilterStatsByKey(const BuildTreeStatsType &stats_in,
                      EventKeyType key,
                      std::vector<EventValueType> &values,
                      bool include_if_present,  // true-> retain only in "values",
                      // false-> retain only not in "values".
                      BuildTreeStatsType *stats_out) {
  KALDI_ASSERT(IsSortedAndUniq(values));
  BuildTreeStatsType::const_iterator iter, end = stats_in.end();
  KALDI_ASSERT(stats_out != NULL);
  stats_out->clear();

  for (iter = stats_in.begin(); iter != end; ++iter) {
    const EventType &evec = iter->first;
    EventValueType val;
    if (! EventMap::Lookup(evec, key, &val)) // no such key. // HERE.
      KALDI_ERR << "SplitStats: key "<< key << " is not present in event vector " << EventTypeToString(evec);
    bool in_values = std::binary_search(values.begin(), values.end(), val);
    if (in_values == include_if_present)
      stats_out->push_back(*iter);
  }
}


Clusterable *SumStats(const BuildTreeStatsType &stats_in) {
  Clusterable *ans = NULL;
  BuildTreeStatsType::const_iterator iter = stats_in.begin(), end = stats_in.end();
  for (; iter != end; ++iter) {
    Clusterable *cl = iter->second;
    if (cl != NULL) {
      if (!ans)  ans = cl->Copy();
      else ans->Add(*cl);
    }
  }
  return ans;
}

BaseFloat SumNormalizer(const BuildTreeStatsType &stats_in) {
  BaseFloat ans = 0.0;
  BuildTreeStatsType::const_iterator iter = stats_in.begin(), end = stats_in.end();
  for (; iter != end; ++iter) {
    Clusterable *cl = iter->second;
    if (cl != NULL) ans += cl->Normalizer();
  }
  return ans;
}

BaseFloat SumObjf(const BuildTreeStatsType &stats_in) {
  BaseFloat ans = 0.0;
  BuildTreeStatsType::const_iterator iter = stats_in.begin(), end = stats_in.end();
  for (; iter != end; ++iter) {
    Clusterable *cl = iter->second;
    if (cl != NULL) ans += cl->Objf();
  }
  return ans;
}


void SumStatsVec(const std::vector<BuildTreeStatsType> &stats_in, std::vector<Clusterable*> *stats_out) {
  KALDI_ASSERT(stats_out != NULL && stats_out->empty());
  stats_out->resize(stats_in.size(), NULL);
  for (size_t i = 0;i < stats_in.size();i++) (*stats_out)[i] = SumStats(stats_in[i]);
}

BaseFloat ObjfGivenMap(const BuildTreeStatsType &stats_in, const EventMap &e) {
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats_in, e, &split_stats);
  std::vector<Clusterable*> summed_stats;
  SumStatsVec(split_stats, &summed_stats);
  BaseFloat ans = SumClusterableObjf(summed_stats);
  DeletePointers(&summed_stats);
  return ans;
}

// This function computes the best initial split of these stats [with this key].
// Returns best objf change (>=0).
BaseFloat ComputeInitialSplit(const std::vector<Clusterable*> &summed_stats,
                              const Questions &q_opts, EventKeyType key,
                              std::vector<EventValueType> *yes_set) {
  KALDI_ASSERT(yes_set != NULL);
  yes_set->clear();
  const QuestionsForKey &key_opts = q_opts.GetQuestionsOf(key);

  // "total" needed for optimization in AddToClustersOptimized,
  // and also used to work otu total objf.
  Clusterable *total = SumClusterable(summed_stats);
  if (total == NULL) return 0.0;  // because there were no stats or non-NULL stats.
  BaseFloat unsplit_objf = total->Objf();

  const std::vector<std::vector<EventValueType> > &questions_of_this_key = key_opts.initial_questions;

  int32 best_idx = -1;
  BaseFloat best_objf_change = 0;

  for (size_t i = 0; i < questions_of_this_key.size(); i++) {
    const std::vector<EventValueType> &yes_set = questions_of_this_key[i];
    std::vector<int32> assignments(summed_stats.size(), 0);  // 0 is index of "no".
    std::vector<Clusterable*> clusters(2);  // no and yes clusters.
    for (std::vector<EventValueType>::const_iterator iter = yes_set.begin(); iter != yes_set.end(); iter++) {
      KALDI_ASSERT(*iter>=0);
      if (*iter < (EventValueType)assignments.size()) assignments[*iter] = 1;
    }
    kaldi::AddToClustersOptimized(summed_stats, assignments, *total, &clusters);
    BaseFloat this_objf = SumClusterableObjf(clusters);

    if (this_objf < unsplit_objf- 0.001*std::abs(unsplit_objf)) {  // got worse; should never happen.
      // of course small differences can be caused by roundoff.
      KALDI_WARN << "Objective function got worse when building tree: "<< this_objf << " < " << unsplit_objf;
      KALDI_ASSERT(!(this_objf < unsplit_objf - 0.01*(200 + std::abs(unsplit_objf))));  // do assert on more stringent check.
    }

    BaseFloat this_objf_change = this_objf - unsplit_objf;
    if (this_objf_change > best_objf_change) {
      best_objf_change = this_objf_change;
      best_idx = i;
    }
    DeletePointers(&clusters);
  }
  delete total;
  if (best_idx != -1)
    *yes_set = questions_of_this_key[best_idx];
  return best_objf_change;
}


// returns best delta-objf.
// If key does not exist, returns 0 and sets yes_set_out to empty.
BaseFloat FindBestSplitForKey(const BuildTreeStatsType &stats,
                              const Questions &q_opts,
                              EventKeyType key,
                              std::vector<EventValueType> *yes_set_out) {
  if (stats.size()<=1) return 0.0;  // cannot split if only zero or one instance of stats.
  if (!PossibleValues(key, stats, NULL)) {
    yes_set_out->clear();
    return 0.0;  // Can't split as key not always defined.
  }
  std::vector<Clusterable*> summed_stats;  // indexed by value corresponding to key. owned here.
  {  // compute summed_stats
    std::vector<BuildTreeStatsType> split_stats;
    SplitStatsByKey(stats, key, &split_stats);
    SumStatsVec(split_stats, &summed_stats);
  }

  std::vector<EventValueType> yes_set;
  BaseFloat improvement = ComputeInitialSplit(summed_stats,
                                               q_opts, key, &yes_set);
  // find best basic question.

  std::vector<int32> assignments(summed_stats.size(), 0);  // assigns to "no" (0) by default.
  for (std::vector<EventValueType>::const_iterator iter = yes_set.begin(); iter != yes_set.end(); iter++) {
    KALDI_ASSERT(*iter>=0);
    if (*iter < (EventValueType)assignments.size()) {
      // this guard necessary in case stats did not have all the
      // values present in "yes_set".
      assignments[*iter] = 1;  // assign to "yes" (1).
    }
  }
  std::vector<Clusterable*> clusters(2, (Clusterable*)NULL);  // no, yes.
  kaldi::AddToClusters(summed_stats, assignments, &clusters);

  EnsureClusterableVectorNotNull(&summed_stats);
  EnsureClusterableVectorNotNull(&clusters);

  // even if improvement == 0 we continue; if we do RefineClusters we may get further improvement.
  // now do the RefineClusters stuff.  Note that this is null-op if
  // q_opts.GetQuestionsOf(key).refine_opts.num_iters == 0.  We could check for this but don't bother;
  // it happens in RefineClusters anyway.

  if (q_opts.GetQuestionsOf(key).refine_opts.num_iters > 0) {
    // If we want to refine the questions... (a bit like k-means w/ 2 classes).
    // Note: the only reason we introduced the if-statement is so the yes_set
    // doesn't get modified (truncated, actually) if we do the refine stuff with
    // zero iters.
    BaseFloat refine_impr = RefineClusters(summed_stats, &clusters, &assignments,
                                           q_opts.GetQuestionsOf(key).refine_opts);
    KALDI_ASSERT(refine_impr > std::min(-1.0, -0.1*fabs(improvement)));
    // refine_impr should always be positive
    improvement += refine_impr;
    yes_set.clear();
    for (size_t i = 0;i < assignments.size();i++) if (assignments[i] == 1) yes_set.push_back(i);
  }
  *yes_set_out = yes_set;
    
  DeletePointers(&clusters);
#ifdef KALDI_PARANOID
  {  // Check the "ans" is correct.
    KALDI_ASSERT(clusters.size() == 2 && clusters[0] == 0 && clusters[1] == 0);
    AddToClusters(summed_stats, assignments, &clusters);
    BaseFloat impr;
    if (clusters[0] == NULL || clusters[1] == NULL) impr = 0.0;
    else impr = clusters[0]->Distance(*(clusters[1]));
    if (!ApproxEqual(impr, improvement) && fabs(impr-improvement) > 0.01) {
      KALDI_WARN << "FindBestSplitForKey: improvements do not agree: "<< impr
                 << " vs. " << improvement;
    }
    DeletePointers(&clusters);
  }
#endif
  DeletePointers(&summed_stats);
  return improvement; // objective-function improvement.
}



/*
  DecisionTreeBuilder is a class used in SplitDecisionTree
*/

class DecisionTreeSplitter {
 public:
  EventMap *GetMap() {
    if (!yes_) {  // leaf.
      return new ConstantEventMap(leaf_);
    } else {
      return new SplitEventMap(key_, yes_set_, yes_->GetMap(), no_->GetMap());
    }
  }
  BaseFloat BestSplit() { return best_split_impr_; } // returns objf improvement (>=0) of best possible split.
  void DoSplit(int32 *next_leaf) {
    if (!yes_) {  // not already split; we are a leaf, so split.
      DoSplitInternal(next_leaf);
    } else {  // find which of our children is best to split, and split that.
      (yes_->BestSplit() >= no_->BestSplit() ? yes_ : no_)->DoSplit(next_leaf);
      best_split_impr_ = std::max(yes_->BestSplit(), no_->BestSplit());  // may have changed.
    }
  }
  DecisionTreeSplitter(EventAnswerType leaf, const BuildTreeStatsType &stats,
                      const Questions &q_opts): q_opts_(q_opts), yes_(NULL), no_(NULL), leaf_(leaf), stats_(stats) {
    // not, this must work when stats is empty too. [just gives zero improvement, non-splittable].
    FindBestSplit();
  }
  ~DecisionTreeSplitter() {
    if (yes_) delete yes_;
    if (no_) delete no_;
  }
 private:
  void DoSplitInternal(int32 *next_leaf) {
    // Does the split; applicable only to leaf nodes.
    KALDI_ASSERT(!yes_);  // make sure children not already set up.
    KALDI_ASSERT(best_split_impr_ > 0);
    EventAnswerType yes_leaf = leaf_, no_leaf = (*next_leaf)++;
    leaf_ = -1;  // we now have no leaf.
    // Now split the stats.
    BuildTreeStatsType yes_stats, no_stats;
    yes_stats.reserve(stats_.size()); no_stats.reserve(stats_.size());  //  probably better than multiple resizings.
    for (BuildTreeStatsType::const_iterator iter = stats_.begin(); iter != stats_.end(); ++iter) {
      const EventType &vec = iter->first;
      EventValueType val;
      if (!EventMap::Lookup(vec, key_, &val)) KALDI_ERR << "DoSplitInternal: key has no value.";
      if (std::binary_search(yes_set_.begin(), yes_set_.end(), val)) yes_stats.push_back(*iter);
      else no_stats.push_back(*iter);
    }
#ifdef KALDI_PARANOID
    {  // Check objf improvement.
      Clusterable *yes_clust = SumStats(yes_stats), *no_clust = SumStats(no_stats);
      BaseFloat impr_check = yes_clust->Distance(*no_clust);
      // this is a negated objf improvement from merging (== objf improvement from splitting).
      if (!ApproxEqual(impr_check, best_split_impr_, 0.01)) {
        KALDI_WARN << "DoSplitInternal: possible problem: "<< impr_check << " != " << best_split_impr_;
      }
      delete yes_clust; delete no_clust;
    }
#endif
    yes_ = new DecisionTreeSplitter(yes_leaf, yes_stats, q_opts_);
    no_ = new DecisionTreeSplitter(no_leaf, no_stats, q_opts_);
    best_split_impr_ = std::max(yes_->BestSplit(), no_->BestSplit());
    stats_.clear();  // note: pointers in stats_ were not owned here.
  }
  void FindBestSplit() {
    // This sets best_split_impr_, key_ and yes_set_.
    // May just pick best question, or may iterate a bit (depends on
    // q_opts; see FindBestSplitForKey for details)
    std::vector<EventKeyType> all_keys;
    q_opts_.GetKeysWithQuestions(&all_keys);
    if (all_keys.size() == 0) {
      KALDI_WARN << "DecisionTreeSplitter::FindBestSplit(), no keys available to split on (maybe no key covered all of your events, or there was a problem with your questions configuration?)";
    }
    best_split_impr_ = 0;
    for (size_t i = 0; i < all_keys.size(); i++) {
      if (q_opts_.HasQuestionsForKey(all_keys[i])) {
        std::vector<EventValueType> temp_yes_set;
        BaseFloat split_improvement = FindBestSplitForKey(stats_, q_opts_, all_keys[i], &temp_yes_set);
        if (split_improvement > best_split_impr_) {
          best_split_impr_ = split_improvement;
          yes_set_ = temp_yes_set;
          key_ = all_keys[i];
        }
      }
    }
  }



  // Data members... Always used:
  const Questions &q_opts_;
  BaseFloat best_split_impr_;

  // If already split:
  DecisionTreeSplitter *yes_;
  DecisionTreeSplitter *no_;

  // Otherwise:
  EventAnswerType leaf_;
  BuildTreeStatsType stats_;  // vector of stats.  pointers inside there not owned here.

  // key and "yes set" of best split:
  EventKeyType key_;
  std::vector<EventValueType> yes_set_;

};

EventMap *SplitDecisionTree(const EventMap &input_map,
                            const BuildTreeStatsType &stats,
                            Questions &q_opts,
                            BaseFloat thresh,
                            int32 max_leaves,  // max_leaves<=0 -> no maximum.
                            int32 *num_leaves,
                            BaseFloat *obj_impr_out,
                            BaseFloat *smallest_split_change_out) {
  KALDI_ASSERT(num_leaves != NULL && *num_leaves > 0);  // can't be 0 or input_map would be empty.
  int32 num_empty_leaves = 0;
  BaseFloat like_impr = 0.0;
  BaseFloat smallest_split_change = 1.0e+20;
  std::vector<DecisionTreeSplitter*> builders;
  {  // set up "builders" [one for each current leaf].  This array is never extended.
    // the structures generated during splitting remain as trees at each array location.
    std::vector<BuildTreeStatsType> split_stats;
    SplitStatsByMap(stats, input_map, &split_stats);
    KALDI_ASSERT(split_stats.size() != 0);
    builders.resize(split_stats.size());  // size == #leaves.
    for (size_t i = 0;i < split_stats.size();i++) {
      EventAnswerType leaf = static_cast<EventAnswerType>(i);
      if (split_stats[i].size() == 0) num_empty_leaves++;
      builders[i] = new DecisionTreeSplitter(leaf, split_stats[i], q_opts);
    }
  }

  {  // Do the splitting.
    int32 count = 0;
    std::priority_queue<std::pair<BaseFloat, size_t> > queue;  // use size_t because logically these
    // are just indexes into the array, not leaf-ids (after splitting they are no longer leaf id's).
    // Initialize queue.
    for (size_t i = 0; i < builders.size(); i++)
      queue.push(std::make_pair(builders[i]->BestSplit(), i));
    // Note-- queue's size never changes from now.  All the alternatives leaves to split are
    // inside the "DecisionTreeSplitter*" objects, in a tree structure.
    while (queue.top().first > thresh
          && (max_leaves<=0 || *num_leaves < max_leaves)) {
      smallest_split_change = std::min(smallest_split_change, queue.top().first);
      size_t i = queue.top().second;
      like_impr += queue.top().first;
      builders[i]->DoSplit(num_leaves);
      queue.pop();
      queue.push(std::make_pair(builders[i]->BestSplit(), i));
      count++;
    }
    KALDI_LOG << "DoDecisionTreeSplit: split "<< count << " times, #leaves now " << (*num_leaves);
  }

  if (smallest_split_change_out)
    *smallest_split_change_out = smallest_split_change;

  EventMap *answer = NULL;

  {  // Create the output EventMap.
    std::vector<EventMap*> sub_trees(builders.size());
    for (size_t i = 0; i < sub_trees.size();i++) sub_trees[i] = builders[i]->GetMap();
    answer = input_map.Copy(sub_trees);
    for (size_t i = 0; i < sub_trees.size();i++) delete sub_trees[i];
  }
  // Free up memory.
  for (size_t i = 0;i < builders.size();i++) delete builders[i];

  if (obj_impr_out != NULL) *obj_impr_out = like_impr;
  return answer;
}


int ClusterEventMapGetMapping(const EventMap &e_in, const BuildTreeStatsType &stats, BaseFloat thresh, std::vector<EventMap*> *mapping) {
  // First map stats
  KALDI_ASSERT(stats.size() != 0);
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, e_in, &split_stats);
  std::vector<Clusterable*> summed_stats;
  SumStatsVec(split_stats, &summed_stats);

  std::vector<int32> indexes;
  std::vector<Clusterable*> summed_stats_contiguous;
  size_t max_index = 0;
  for (size_t i = 0;i < summed_stats.size();i++) {
    if (summed_stats[i] != NULL) {
      indexes.push_back(i);
      summed_stats_contiguous.push_back(summed_stats[i]);
      if (i > max_index) max_index = i;
    }
  }
  if (summed_stats_contiguous.empty()) {
    KALDI_WARN << "ClusterBottomUp: nothing to cluster.\n";
    return 0;  // nothing merged.
  }

  std::vector<int32> assignments;
  BaseFloat normalizer = SumClusterableNormalizer(summed_stats_contiguous),
      change;
  change = ClusterBottomUp(summed_stats_contiguous,
                           thresh,
                           0,  // no min-clust: use threshold for now.
                           NULL,  // don't need clusters out.
                           &assignments);  // this algorithm is quadratic, so might be quite slow.


  KALDI_ASSERT(assignments.size() == summed_stats_contiguous.size() && !assignments.empty());
  size_t num_clust = * std::max_element(assignments.begin(), assignments.end()) + 1;
  int32 num_combined = summed_stats_contiguous.size() - num_clust;
  KALDI_ASSERT(num_combined >= 0);

  KALDI_VLOG(2) <<  "ClusterBottomUp combined "<< num_combined << " leaves and gave a likelihood change of " << change << ", normalized = " << (change/normalizer) << ", normalizer = " << normalizer;
  KALDI_ASSERT(change < 0.0001);  // should be negative or zero.

  KALDI_ASSERT(mapping != NULL);
  if (max_index >= mapping->size()) mapping->resize(max_index+1, NULL);

  for (size_t i = 0;i < summed_stats_contiguous.size();i++) {
    size_t index = indexes[i];
    size_t new_index = indexes[assignments[i]];  // index assigned by clusterig-- map to existing indices in the map,
    // that we clustered from, so we don't conflict with indices in other parts of the tree.
    KALDI_ASSERT((*mapping)[index] == NULL || "Error: Cluster seems to have been called for different parts of the tree with overlapping sets of indices.");
    (*mapping)[index] = new ConstantEventMap(new_index);
  }
  DeletePointers(&summed_stats);
  return num_combined;
}


EventMap *RenumberEventMap(const EventMap &e_in, int32 *num_leaves) {
  EventType empty_vec;
  std::vector<EventAnswerType> initial_leaves;  // before renumbering.
  e_in.MultiMap(empty_vec, &initial_leaves);
  if (initial_leaves.empty()) {
    KALDI_ASSERT(num_leaves);
    if (num_leaves) *num_leaves = 0;
    return e_in.Copy();
  }
  SortAndUniq(&initial_leaves);
  EventAnswerType max_leaf_plus_one = initial_leaves.back() + 1;  // will typically, but not always, equal *num_leaves.
  std::vector<EventMap*> mapping(max_leaf_plus_one, (EventMap*)NULL);
  std::vector<EventAnswerType>::iterator iter = initial_leaves.begin(), end = initial_leaves.end();
  EventAnswerType cur_leaf = 0;
  for (; iter != end; ++iter) {
    KALDI_ASSERT(*iter >= 0 && *iter<max_leaf_plus_one);
    mapping[*iter] = new ConstantEventMap(cur_leaf++);
  }
  EventMap *ans = e_in.Copy(mapping);
  DeletePointers(&mapping);
  KALDI_ASSERT((size_t)cur_leaf == initial_leaves.size());
  if (num_leaves) *num_leaves = cur_leaf;
  return ans;
}

EventMap *MapEventMapLeaves(const EventMap &e_in,
                            const std::vector<int32> &mapping_in) {
  std::vector<EventMap*> mapping(mapping_in.size());
  for (size_t i = 0; i < mapping_in.size(); i++)
    mapping[i] = new ConstantEventMap(mapping_in[i]);
  EventMap *ans = e_in.Copy(mapping);
  DeletePointers(&mapping);
  return ans;
}

EventMap *ClusterEventMap(const EventMap &e_in, const BuildTreeStatsType &stats,
                          BaseFloat thresh, int32 *num_removed_ptr) {
  std::vector<EventMap*> mapping;
  int32 num_removed = ClusterEventMapGetMapping(e_in, stats, thresh, &mapping);
  EventMap *ans = e_in.Copy(mapping);
  DeletePointers(&mapping);
  if (num_removed_ptr != NULL) *num_removed_ptr = num_removed;
  return ans;

}

EventMap *ShareEventMapLeaves(const EventMap &e_in, EventKeyType key,
                              std::vector<std::vector<EventValueType> > &values,
                              int32 *num_leaves) {
  // the use of "pdfs" as the name of the next variable reflects the anticipated
  // use of this function.
  std::vector<std::vector<EventAnswerType> > pdfs(values.size());
  for (size_t i = 0; i < values.size(); i++) {
    EventType evec;
    for (size_t j = 0; j < values[i].size(); j++) {
      evec.push_back(MakeEventPair(key, values[i][j]));
      size_t size_at_start = pdfs[i].size();
      e_in.MultiMap(evec, &(pdfs[i]));  // append any corresponding pdfs to pdfs[i].
      if (pdfs[i].size() == size_at_start) {  // Nothing added... unexpected.
        KALDI_WARN << "ShareEventMapLeaves: had no leaves for key = " << key
                   << ", value = " << (values[i][j]);
      }
    }
    SortAndUniq(&(pdfs[i]));
  }
  std::vector<EventMap*> remapping;
  for (size_t i = 0; i < values.size(); i++) {
    if (pdfs[i].empty())
      KALDI_WARN << "ShareEventMapLeaves: no leaves in one bucket.";  // not expected.
    else {
      EventAnswerType map_to_this = pdfs[i][0];  // map all in this bucket
      // to this value.
      for (size_t j = 1; j < pdfs[i].size(); j++) {
        EventAnswerType leaf = pdfs[i][j];
        KALDI_ASSERT(leaf>=0);
        if (remapping.size() <= static_cast<size_t>(leaf))
          remapping.resize(leaf+1, NULL);
        KALDI_ASSERT(remapping[leaf] == NULL);
        remapping[leaf] = new ConstantEventMap(map_to_this);
      }
    }
  }
  EventMap *shared = e_in.Copy(remapping);
  DeletePointers(&remapping);
  EventMap *renumbered = RenumberEventMap(*shared, num_leaves);
  delete shared;
  return renumbered;
}


void DeleteBuildTreeStats(BuildTreeStatsType *stats) {
  KALDI_ASSERT(stats != NULL);
  BuildTreeStatsType::iterator iter = stats->begin(), end = stats->end();
  for (; iter!= end; ++iter) if (iter->second != NULL) { delete iter->second; iter->second = NULL; } // set to NULL for extra safety.
}

EventMap *GetToLengthMap(const BuildTreeStatsType &stats, int32 P,
                         const std::vector<EventValueType> *phones,
                         int32 default_length) {
  std::vector<BuildTreeStatsType> stats_by_phone;
  try {
    SplitStatsByKey(stats, P, &stats_by_phone);
  } catch(const std::runtime_error &err) {
    KALDI_ERR << "Caught exception in GetToLengthMap: you seem "
        "to have provided invalid stats [no central-phone "
        "key].  Message was: " << err.what();
  }
  std::map<EventValueType, EventAnswerType> phone_to_length;
  for (size_t p = 0; p < stats_by_phone.size(); p++) {
    if (! stats_by_phone[p].empty()) {
      std::vector<BuildTreeStatsType> stats_by_length;
      try {
        SplitStatsByKey(stats_by_phone[p], kPdfClass, &stats_by_length);
      } catch(const std::runtime_error &err) {
        KALDI_ERR << "Caught exception in GetToLengthMap: you seem "
            "to have provided invalid stats [no position "
            "key].  Message was: " << err.what();
      }
      size_t length = stats_by_length.size();
      for (size_t i = 0; i < length; i++) {
        if (stats_by_length[i].empty()) {
          KALDI_ERR << "There are no stats available for position " << i
                    << " of phone " << p;
        }
      }
      phone_to_length[p] = length;
    }
  }
  if (phones != NULL) {  // set default length for unseen phones.
    for (size_t i = 0; i < phones->size(); i++) {
      if (phone_to_length.count( (*phones)[i] ) == 0) {  // unseen.
        phone_to_length[(*phones)[i]] = default_length;
      }
    }
  }
  EventMap *ans = new TableEventMap(P, phone_to_length);
  return ans;
}

// Recursive routine that is a helper to ClusterEventMapRestricted.
// returns number removed.
static int32 ClusterEventMapRestrictedHelper(const EventMap &e_in,
                                             const BuildTreeStatsType &stats,
                                             BaseFloat thresh,
                                             std::vector<EventKeyType> keys,
                                             std::vector<EventMap*> *leaf_mapping) {
  if (keys.size() == 0) {
    return ClusterEventMapGetMapping(e_in, stats, thresh, leaf_mapping);
  } else {  // split on the key.
    int32 ans = 0;
    std::vector<BuildTreeStatsType> split_stats;
    SplitStatsByKey(stats, keys.back(), &split_stats);
    keys.pop_back();
    for (size_t i = 0; i< split_stats.size(); i++)
      if (split_stats[i].size() != 0)
        ans += ClusterEventMapRestrictedHelper(e_in, split_stats[i], thresh, keys, leaf_mapping);
    return ans;
  }
}

EventMap *ClusterEventMapRestrictedByKeys(const EventMap &e_in,
                                          const BuildTreeStatsType &stats,
                                          BaseFloat thresh,
                                          const std::vector<EventKeyType> &keys,
                                          int32 *num_removed) {
  std::vector<EventMap*> leaf_mapping;  // For output of ClusterEventMapGetMapping.

  int32 nr = ClusterEventMapRestrictedHelper(e_in, stats, thresh, keys, &leaf_mapping);
  if (num_removed != NULL) *num_removed = nr;

  EventMap *ans = e_in.Copy(leaf_mapping);
  DeletePointers(&leaf_mapping);
  return ans;
}


EventMap *ClusterEventMapRestrictedByMap(const EventMap &e_in,
                                         const BuildTreeStatsType &stats,
                                         BaseFloat thresh,
                                         const EventMap &e_restrict,
                                         int32 *num_removed_ptr) {
  std::vector<EventMap*> leaf_mapping;

  std::vector<BuildTreeStatsType> split_stats;
  int num_removed = 0;
  SplitStatsByMap(stats, e_restrict, &split_stats);
  for (size_t i = 0; i < split_stats.size(); i++) {
    if (!split_stats[i].empty())
      num_removed += ClusterEventMapGetMapping(e_in, split_stats[i], thresh,
                                               &leaf_mapping);
  }

  if (num_removed_ptr != NULL) *num_removed_ptr = num_removed;

  EventMap *ans = e_in.Copy(leaf_mapping);
  DeletePointers(&leaf_mapping);
  return ans;
}



EventMap *GetStubMap(int32 P,
                     const std::vector<std::vector<int32> > &phone_sets,
                     const std::vector<int32> &phone2num_pdf_classes,
                     const std::vector<bool> &share_roots,
                     int32 *num_leaves_out) {

  {  // Checking inputs.
    KALDI_ASSERT(!phone_sets.empty() && share_roots.size() == phone_sets.size());
    std::set<int32> all_phones;
    for (size_t i = 0; i < phone_sets.size(); i++) {
      KALDI_ASSERT(IsSortedAndUniq(phone_sets[i]));
      KALDI_ASSERT(!phone_sets[i].empty());
      for (size_t j = 0; j < phone_sets[i].size(); j++) {
        KALDI_ASSERT(all_phones.count(phone_sets[i][j]) == 0);  // check not present.
        all_phones.insert(phone_sets[i][j]);
      }
    }
  }

  // Initially create a single leaf for each phone set.

  size_t max_set_size = 0;
  int32 highest_numbered_phone = 0;
  for (size_t i = 0; i < phone_sets.size(); i++) {
    max_set_size = std::max(max_set_size, phone_sets[i].size());
    highest_numbered_phone =
        std::max(highest_numbered_phone,
                 * std::max_element(phone_sets[i].begin(), phone_sets[i].end()));
  }

  if (phone_sets.size() == 1) {  // there is only one set so the recursion finishes.
    if (share_roots[0]) {  // if "shared roots" return a single leaf.
      return new ConstantEventMap( (*num_leaves_out)++ );
    } else {  // not sharing roots -> work out the length and return a
             // TableEventMap splitting on length.
      EventAnswerType max_len = 0;
      for (size_t i = 0; i < phone_sets[0].size(); i++) {
        EventAnswerType len;
        EventValueType phone = phone_sets[0][i];
        KALDI_ASSERT(static_cast<size_t>(phone) < phone2num_pdf_classes.size());
        len = phone2num_pdf_classes[phone];
        KALDI_ASSERT(len > 0);
        if (i == 0) max_len = len;
        else {
          if (len != max_len) {
            KALDI_WARN << "Mismatching lengths within a phone set: " << len
                       << " vs. " << max_len << " [unusual, but not necessarily fatal]. ";
            max_len = std::max(len, max_len);
          }
        }
      }
      std::map<EventValueType, EventAnswerType> m;
      for (EventAnswerType p = 0; p < max_len; p++)
        m[p] = (*num_leaves_out)++;
      return new TableEventMap(kPdfClass,  // split on hmm-position
                               m);
    }
  } else if (max_set_size == 1
            && static_cast<int32>(phone_sets.size()) <= 2*highest_numbered_phone) {
    // create table map splitting on phone-- more efficient.
    // the part after the && checks that this would not contain a very sparse vector.
    std::map<EventValueType, EventMap*> m;
    for (size_t i = 0; i < phone_sets.size(); i++) {
      std::vector<std::vector<int32> > phone_sets_tmp;
      phone_sets_tmp.push_back(phone_sets[i]);
      std::vector<bool> share_roots_tmp;
      share_roots_tmp.push_back(share_roots[i]);
      EventMap *this_stub = GetStubMap(P, phone_sets_tmp, phone2num_pdf_classes,
                                       share_roots_tmp,
                                       num_leaves_out);
      KALDI_ASSERT(m.count(phone_sets_tmp[0][0]) == 0);
      m[phone_sets_tmp[0][0]] = this_stub;
    }
    return new TableEventMap(P, m);
  } else {
    // Do a split.  Recurse.
    size_t half_sz = phone_sets.size() / 2;
    std::vector<std::vector<int32> >::const_iterator half_phones =
        phone_sets.begin() + half_sz;  
    std::vector<bool>::const_iterator half_share =
        share_roots.begin() + half_sz;
    std::vector<std::vector<int32> > phone_sets_1, phone_sets_2;
    std::vector<bool> share_roots_1, share_roots_2;
    phone_sets_1.insert(phone_sets_1.end(), phone_sets.begin(), half_phones);
    phone_sets_2.insert(phone_sets_2.end(), half_phones, phone_sets.end());
    share_roots_1.insert(share_roots_1.end(), share_roots.begin(), half_share);
    share_roots_2.insert(share_roots_2.end(), half_share, share_roots.end());

    EventMap *map1 = GetStubMap(P, phone_sets_1, phone2num_pdf_classes, share_roots_1, num_leaves_out);
    EventMap *map2 = GetStubMap(P, phone_sets_2, phone2num_pdf_classes, share_roots_2, num_leaves_out);

    std::vector<EventKeyType> all_in_first_set;
    for (size_t i = 0; i < half_sz; i++)
      for (size_t j = 0; j < phone_sets[i].size(); j++)
        all_in_first_set.push_back(phone_sets[i][j]);
    std::sort(all_in_first_set.begin(), all_in_first_set.end());
    KALDI_ASSERT(IsSortedAndUniq(all_in_first_set));
    return new SplitEventMap(P, all_in_first_set, map1, map2);
  }
}

// convert stats to different (possibly smaller) context-window size.
bool ConvertStats(int32 oldN, int32 oldP, int32 newN, int32 newP,
                  BuildTreeStatsType *stats) {
  bool warned = false;
  KALDI_ASSERT(stats != NULL && oldN > 0 && newN > 0 && oldP >= 0
               && newP >= 0 && newP < newN && oldP < oldN);
  if (newN > oldN) {  // can't add unseen context.
    KALDI_WARN << "Cannot convert stats to larger context: " << newN
               << " > " << oldN;
    return false;
  }
  if (newP > oldP) {
    KALDI_WARN << "Cannot convert stats to have more left-context: " << newP
               << " > " << oldP;
  }
  if (newN-newP-1 > oldN-oldP-1) {
    KALDI_WARN << "Cannot convert stats to have more right-context: " << (newN-newP-1)
               << " > " << (oldN-oldP-1);
  }
  // if shift < 0, this is by how much we "shift down" key
  // values.
  int32 shift = newP - oldP;  // shift <= 0.

  for (size_t i = 0; i < stats->size(); i++) {
    EventType &evec = (*stats)[i].first;
    EventType evec_new;
    for (size_t j = 0; j < evec.size(); j++) {
      EventKeyType key = evec[j].first;
      if (key >= 0 && key < oldN) {  //
        key += shift;
        if (key >= 0 && key < newN) // within new context window...
          evec_new.push_back(std::make_pair(key, evec[j].second));
      } else {
        if (key != -1) {
          // don't understand this key value but assume for now
          // it's something that doesn't interact with the context window.
          if (!warned) {
            KALDI_WARN << "Stats had keys defined that we cannot interpret";
            warned = true;
          }
        }
        evec_new.push_back(evec[j]);
      }
    }
    evec = evec_new;  // Assign the modified EventVector with possibly
    // deleted keys.
  }
  return true;
}


} // end namespace kaldi

