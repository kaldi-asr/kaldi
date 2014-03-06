// tree/build-tree-utils-test.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "util/stl-utils.h"
#include "util/kaldi-io.h"
#include "tree/build-tree-utils.h"
#include "tree/clusterable-classes.h"


namespace kaldi {

void TestTrivialTree() {
  int32 nleaves = 0;
  EventMap *tree = TrivialTree(&nleaves);
  EventType empty; EventAnswerType ans;
  bool b = tree->Map(empty, &ans);
  KALDI_ASSERT(b);
  KALDI_ASSERT(nleaves == 1);
  delete tree;
}

void TestPossibleValues() {
  BuildTreeStatsType stats;
  std::map<EventKeyType, std::set<EventValueType> > all_key_vals;
  for (size_t i = 0;i < 20;i++) {
    std::map<EventKeyType, EventValueType> key_vals;
    for (size_t j = 0;j < 20;j++) {
      EventKeyType k = RandInt(-5, 10);
      EventValueType v = RandInt(0, 10);
      if (key_vals.count(k) == 0) {
        key_vals[ k ] =  v;
        all_key_vals[k].insert(v);
      }
    }
    EventType evec;
    CopyMapToVector(key_vals, &evec);
    stats.push_back(std::pair<EventType, Clusterable*>(evec, (Clusterable*)NULL));
  }
  for (std::map<EventKeyType, std::set<EventValueType> >::iterator iter = all_key_vals.begin();
      iter != all_key_vals.end(); iter++) {
    EventKeyType key = iter->first;
    std::vector<EventValueType> vals1, vals2;
    CopySetToVector(iter->second, &vals1);
    PossibleValues(key, stats, &vals2);
    if (vals1 != vals2) {
      printf("vals differ!\n");
      for (size_t i = 0;i < vals1.size();i++) std::cout << vals1[i] << " ";
      std::cout<<'\n';
      for (size_t i = 0;i < vals2.size();i++) std::cout << vals2[i] << " ";
      std::cout<<'\n';
      KALDI_ASSERT(0);
    }
  }
}

void TestConvertStats() {
  {
    BuildTreeStatsType stats;
    EventType evec;
    // this example is of ctx window (10, 11, 12), and pdf-class = 1.
    evec.push_back(std::pair<int32, int32>(-1, 1));
    evec.push_back(std::pair<int32, int32>(0, 10));
    evec.push_back(std::pair<int32, int32>(1, 11));
    evec.push_back(std::pair<int32, int32>(2, 12));
    stats.push_back(std::make_pair(evec, static_cast<Clusterable*>(NULL)));
    int32 oldN = 3, oldP = 1, newN = 1, newP = 0;
    ConvertStats(oldN, oldP, newN, newP, &stats);
    EventType new_evec = stats[0].first;
    KALDI_ASSERT(new_evec.size() == 2);  // keys: -1, 0.
    KALDI_ASSERT(new_evec[0].first == -1 && new_evec[0].second == 1);
    KALDI_ASSERT(new_evec[1].first == 0 && new_evec[1].second == 11);
  }

  {  // as above, but convert to left bigram (N = 2, P = 1).
    BuildTreeStatsType stats;
    EventType evec;
    // this example is of ctx window (10, 11, 12), and pdf-class = 1.
    evec.push_back(std::pair<int32, int32>(-1, 1));
    evec.push_back(std::pair<int32, int32>(0, 10));
    evec.push_back(std::pair<int32, int32>(1, 11));
    evec.push_back(std::pair<int32, int32>(2, 12));
    stats.push_back(std::make_pair(evec, static_cast<Clusterable*>(NULL)));
    int32 oldN = 3, oldP = 1, newN = 2, newP = 1;
    ConvertStats(oldN, oldP, newN, newP, &stats);
    EventType new_evec = stats[0].first;
    KALDI_ASSERT(new_evec.size() == 3);  // keys: -1, 0, 1.
    KALDI_ASSERT(new_evec[0].first == -1 && new_evec[0].second == 1);
    KALDI_ASSERT(new_evec[1].first == 0 && new_evec[1].second == 10);
    KALDI_ASSERT(new_evec[2].first == 1 && new_evec[2].second == 11);
  }

  {  // as above, but leave unchanged.
    BuildTreeStatsType stats;
    EventType evec;
    // this example is of ctx window (10, 11, 12), and pdf-class = 1.
    evec.push_back(std::pair<int32, int32>(-1, 1));
    evec.push_back(std::pair<int32, int32>(0, 10));
    evec.push_back(std::pair<int32, int32>(1, 11));
    evec.push_back(std::pair<int32, int32>(2, 12));
    stats.push_back(std::make_pair(evec, static_cast<Clusterable*>(NULL)));
    int32 oldN = 3, oldP = 1, newN = 3, newP = 1;
    ConvertStats(oldN, oldP, newN, newP, &stats);
    EventType new_evec = stats[0].first;
    KALDI_ASSERT(new_evec == evec);
  }
}
void TestSplitStatsByKey() {
  {
    BuildTreeStatsType stats;
    for(int32 i = 0; i < 100; i++) {
      EventType evec;
      if (rand() % 2)
        evec.push_back(std::make_pair(12, rand() % 10));
      evec.push_back(std::make_pair(10, rand() % 10));
      if (rand() % 2)
        evec.push_back(std::make_pair(8, rand() % 10));
      std::sort(evec.begin(), evec.end());
      stats.push_back(std::make_pair(evec, static_cast<Clusterable*>(NULL)));
    }
    std::vector<BuildTreeStatsType> stats_vec;
    SplitStatsByKey(stats, 10, &stats_vec);
    for(int32 i = 0; i < static_cast<int32>(stats_vec.size()); i++) {
      for(int32 j = 0; j < static_cast<int32>(stats_vec[i].size()); j++) {
        EventAnswerType ans;
        bool ok = EventMap::Lookup(stats_vec[i][j].first, 10, &ans);
        KALDI_ASSERT(ok && ans == i);
      }
    }
  }
}

void TestFindAllKeys() {
  for (size_t iter = 0;iter < 10;iter++) {
    BuildTreeStatsType stats;
    std::set<EventKeyType> all_keys_union;
    std::set<EventKeyType> all_keys_intersection;

    for (size_t i = 0;i < 3;i++) {
      std::map<EventKeyType, EventValueType> key_vals;
      for (size_t j = 0;j < 5;j++) {
        EventKeyType k = RandInt(-2, 1);
        EventValueType v = RandInt(0, 10);
        key_vals[k] =  v;
      }
      EventType evec;
      CopyMapToVector(key_vals, &evec);
      stats.push_back(std::pair<EventType, Clusterable*>(evec, (Clusterable*) NULL));
      std::set<EventKeyType> s;
      CopyMapKeysToSet(key_vals, &s);
      if (i == 0) { all_keys_union = s; all_keys_intersection = s; }
      else {
        std::set<EventKeyType> new_intersection;
        for (std::set<EventKeyType>::iterator iter = s.begin(); iter != s.end(); iter++) {
          all_keys_union.insert(*iter);
          if (all_keys_intersection.count(*iter) != 0) new_intersection.insert(*iter);
        }
        all_keys_intersection = new_intersection;
      }
    }

    {  // test in union mode.
      std::vector<EventKeyType> keys1, keys2;
      CopySetToVector(all_keys_union, &keys1);
      FindAllKeys(stats, kAllKeysUnion, &keys2);
      KALDI_ASSERT(keys1 == keys2);
    }
    {  // test in intersection mode.
      std::vector<EventKeyType> keys1, keys2;
      CopySetToVector(all_keys_intersection, &keys1);
      FindAllKeys(stats, kAllKeysIntersection, &keys2);
      KALDI_ASSERT(keys1 == keys2);
    }
    {  // test in insist-same mode.
      std::vector<EventKeyType> keys1, keys2;
      CopySetToVector(all_keys_intersection, &keys1);
      try {
        FindAllKeys(stats, kAllKeysInsistIdentical, &keys2);  // note, it SHOULD throw an exception here.
        // This is for testing purposes.  It gets caught.
        KALDI_ASSERT(keys1 == keys2);
        KALDI_ASSERT(all_keys_union == all_keys_intersection);
      } catch(...) {  // it should throw exception if all keys are not the same.
        KALDI_LOG << "Ignore previous error.";
        KALDI_ASSERT(all_keys_union != all_keys_intersection);
      }
    }
  }
}


void TestDoTableSplit() {
  EventValueType numvals = 11;
  for (size_t iter = 0;iter < 10;iter++) {
    BuildTreeStatsType stats;
    int32 nKeys = 3;
    EventKeyType k = RandInt(-1, nKeys-2);  // key we will split on.
    std::set<EventValueType> all_vals;
    for (size_t i = 0;i < 10;i++) {
      EventType evec;
      for (EventKeyType kk = -1;kk < nKeys-1;kk++) {
        EventValueType v = RandInt(0, numvals);
        if (kk == k) all_vals.insert(v);
        evec.push_back(std::make_pair(kk, v));
      }
      stats.push_back(std::pair<EventType, Clusterable*>(evec, (Clusterable*) NULL));
    }
    int32 nleaves = 0;
    EventMap *trivial_map = TrivialTree(&nleaves);

    EventMap *table_map = DoTableSplit(*trivial_map, k, stats, &nleaves);
    KALDI_ASSERT(nleaves <= numvals);
    for (size_t i = 0;i < 10;i++) {
      size_t idx1 = RandInt(0, stats.size()-1), idx2 = RandInt(0, stats.size()-1);
      EventAnswerType ans1;
      table_map->Map(stats[idx1].first, &ans1);
      EventAnswerType ans2;
      table_map->Map(stats[idx2].first, &ans2);

      EventValueType val1, val2;
      bool b = EventMap::Lookup(stats[idx1].first, k, &val1)
             && EventMap::Lookup(stats[idx2].first, k, &val2);
      KALDI_ASSERT(b);
      KALDI_ASSERT(val1 >= 0 );
      KALDI_ASSERT( (val1 == val2) == (ans1 == ans2) );
    }
    for (EventValueType i = 0;i < numvals+1;i++) {
      if (all_vals.count(i) == 0) {
        EventType v; v.push_back(std::make_pair(k, i));
        EventAnswerType ans;
        bool b = table_map->Map(v, &ans);
        KALDI_ASSERT(!b);  // check it maps stuff we never saw to undefined.
      }
    }
    delete trivial_map;
    delete table_map;
  }
}



void TestClusterEventMapGetMappingAndRenumberEventMap() {
  EventKeyType key = 0;  // just one key.
  for (size_t iter = 0;iter < 1;iter++) {  // in loop anyway.
    BuildTreeStatsType stats;
    EventValueType cur_value = 0;
    int32 num_clust = 10;
    for (int32 i = 0;i < num_clust;i++) {  // this will correspond to the "cluster".
      size_t n = 1 + rand() % 3;
      for (size_t j = 0;j < n;j++) {
        BaseFloat scalar = static_cast<BaseFloat>(i) + RandUniform()*0.001;
        EventType evec;
        evec.push_back(std::make_pair(key, cur_value++));
        stats.push_back(std::make_pair(evec, static_cast<Clusterable*>(new ScalarClusterable(scalar))));
      }
    }

    int32 nleaves = 0;
    EventMap *trivial_map = TrivialTree(&nleaves);

    EventMap *table_map = DoTableSplit(*trivial_map, key, stats, &nleaves);
    KALDI_ASSERT(nleaves == cur_value);

    std::vector<EventMap*> mapping;
    int32 num_reduced = ClusterEventMapGetMapping(*table_map, stats, 0.1, &mapping);

    std::cout << "TestCluster(): num_reduced = "<<num_reduced<<", expected: "<<cur_value<<" - "<<num_clust<<" = "<<(cur_value-num_clust)<<'\n';
    KALDI_ASSERT(num_reduced == cur_value - num_clust);

    EventMap *clustered_map = table_map->Copy(mapping);

    EventAnswerType new_nleaves;
    EventMap *renumbered_map = RenumberEventMap(*clustered_map, &new_nleaves);
    KALDI_ASSERT(new_nleaves == num_clust);

    std::vector<EventAnswerType> orig_answers, clustered_answers, renumbered_answers;

    EventType empty_vec;
    table_map->MultiMap(empty_vec, &orig_answers);
    clustered_map->MultiMap(empty_vec, &clustered_answers);
    renumbered_map->MultiMap(empty_vec, &renumbered_answers);

    SortAndUniq(&orig_answers);
    SortAndUniq(&clustered_answers);
    SortAndUniq(&renumbered_answers);
    KALDI_ASSERT(orig_answers.size() == (size_t) cur_value);
    KALDI_ASSERT(clustered_answers.size() == (size_t) num_clust);
    KALDI_ASSERT(renumbered_answers.size() == (size_t) num_clust);
    KALDI_ASSERT(renumbered_map->MaxResult()+1 == num_clust);

    DeletePointers(&mapping);
    delete renumbered_map;
    delete clustered_map;
    delete table_map;
    delete trivial_map;
    DeleteBuildTreeStats(&stats);
  }
}


void TestClusterEventMapGetMappingAndRenumberEventMap2() {
  EventKeyType key = 0;  // just one key.
  for (size_t iter = 0;iter < 1;iter++) {  // in loop anyway.
    BuildTreeStatsType stats;
    BuildTreeStatsType stats_reduced;
    EventValueType cur_value = 0;
    int32 num_clust = 10;
    for (int32 i = 0;i < num_clust;i++) {  // this will correspond to the "cluster".
      size_t n = 1 + rand() % 3;
      for (size_t j = 0;j < n;j++) {
        BaseFloat scalar = static_cast<BaseFloat>(i) + RandUniform()*0.001;
        EventType evec;
        evec.push_back(std::make_pair(key, cur_value++));
        stats.push_back(std::make_pair(evec, static_cast<Clusterable*>(new ScalarClusterable(scalar))));
        if (rand() % 10 < 5) stats_reduced.push_back(stats.back());
      }
    }

    int32 nleaves = 0;
    EventMap *trivial_map = TrivialTree(&nleaves);

    EventMap *table_map = DoTableSplit(*trivial_map, key, stats, &nleaves);
    KALDI_ASSERT(nleaves == cur_value);

    std::vector<EventMap*> mapping;
    int32 num_reduced = ClusterEventMapGetMapping(*table_map, stats_reduced, 0.1, &mapping);

    std::cout << "TestCluster(): num_reduced = "<<num_reduced<<", expected [ignoring gaps]: "<<cur_value<<" - "<<num_clust<<" = "<<(cur_value-num_clust)<<'\n';
    // KALDI_ASSERT(num_reduced == cur_value - num_clust);

    EventMap *clustered_map = table_map->Copy(mapping);

    EventAnswerType new_nleaves;
    EventMap *renumbered_map = RenumberEventMap(*clustered_map, &new_nleaves);
    // KALDI_ASSERT(new_nleaves == num_clust);

    std::vector<EventAnswerType> orig_answers, clustered_answers, renumbered_answers;

    EventType empty_vec;
    table_map->MultiMap(empty_vec, &orig_answers);
    clustered_map->MultiMap(empty_vec, &clustered_answers);
    renumbered_map->MultiMap(empty_vec, &renumbered_answers);

    SortAndUniq(&orig_answers);
    SortAndUniq(&clustered_answers);
    SortAndUniq(&renumbered_answers);
    // KALDI_ASSERT(orig_answers.size() == (size_t) cur_value);
    // KALDI_ASSERT(clustered_answers.size() == (size_t) num_clust);
    // KALDI_ASSERT(renumbered_answers.size() == (size_t) num_clust);
    // KALDI_ASSERT(renumbered_map->MaxResult()+1 == num_clust);

    DeletePointers(&mapping);
    delete renumbered_map;
    delete clustered_map;
    delete table_map;
    delete trivial_map;
    DeleteBuildTreeStats(&stats);
  }
}


void TestClusterEventMap() {
  // This second testing routine checks that ClusterEventMap does not renumber leaves whose stats
  // we exclude.
  // ClusterEventMapGetMapping (the internal version) were tested in another
  // testing routine.

  EventKeyType key = 0;  // just one key.
  for (size_t iter = 0;iter < 1;iter++) {  // in loop anyway.
    BuildTreeStatsType stats;
    EventValueType cur_value = 0;
    int32 num_clust = 10;
    for (int32 i = 0;i < num_clust;i++) {  // this will correspond to the "cluster".
      size_t n = 1 + rand() % 3;
      for (size_t j = 0;j < n;j++) {
        BaseFloat scalar = static_cast<BaseFloat>(i) + RandUniform()*0.001;
        EventType evec;
        evec.push_back(std::make_pair(key, cur_value++));
        stats.push_back(std::make_pair(evec, static_cast<Clusterable*>(new ScalarClusterable(scalar))));
      }
    }

    int32 nleaves = 0;
    EventMap *trivial_map = TrivialTree(&nleaves);

    EventMap *table_map = DoTableSplit(*trivial_map, key, stats, &nleaves);
    KALDI_ASSERT(nleaves == cur_value);

    std::set<EventValueType> exclude_leaves;
    for (size_t i = 0;i < 4;i++) exclude_leaves.insert(rand() % num_clust);
    BuildTreeStatsType stats_excluded;
    BuildTreeStatsType stats_included;
    for (size_t i = 0;i < stats.size();i++) {
      if (exclude_leaves.count(stats[i].first[0].second) != 0) {  // this code relies on the fact that there is just one event in the EventVector.
        stats_excluded.push_back(stats[i]);
      } else {
        stats_included.push_back(stats[i]);
      }
    }
    KALDI_ASSERT(!stats_excluded.empty()&&!stats_included.empty() && stats_excluded.size()+stats_included.size() == stats.size());

    int32 num_reduced;
    EventMap *clustered_map = ClusterEventMap(*table_map, stats_included, 0.1, &num_reduced);

    std::cout << "TestCluster*(): num_reduced = "<<num_reduced<<", expected [without exclusion]: "<<cur_value<<" - "<<num_clust<<" = "<<(cur_value-num_clust)<<'\n';

    // Make sure stats we excluded are not renumbered.
    for (size_t i = 0;i < stats_excluded.size();i++) {
      const EventType &evec = stats_excluded[i].first;
      EventAnswerType ans;  table_map->Map(evec, &ans);
      EventAnswerType  ans2; clustered_map->Map(evec, &ans2);
      KALDI_ASSERT(ans == ans2);
    }

    delete clustered_map;
    delete table_map;
    delete trivial_map;
    DeleteBuildTreeStats(&stats);
  }
}


void TestClusterEventMapRestricted() {

  // TestClusterEventMapRestricted() tests that ClusterEventMapRestricted()
  // does not combine leaves that we were trying to keep separate.

  bool test_by_key = (rand()%2 == 0);

  int32 num_keys = 1 + rand() % 4;
  std::vector<EventKeyType> keys;

  {  // randomly choose keys.  Will always define all of them.
    std::set<EventKeyType> keys_set;
    while (keys_set.size() < (size_t)num_keys)
      keys_set.insert(  (rand() % (num_keys + 10)) - 3 );
    CopySetToVector(keys_set, &keys);
  }


  BuildTreeStatsType stats;

  int32 n_stats = 1 + (rand() % 10);
  n_stats *= n_stats;  // up to 81 stats.

  for (size_t i = 0; i < (size_t)n_stats; i++) {
    EventType evec;

    for (size_t j = 0; j < keys.size(); j++) {
      EventValueType val = rand() % 100;
      evec.push_back(std::make_pair(keys[j], val));
    }
    stats.push_back(std::make_pair(evec, static_cast<Clusterable*>(new ScalarClusterable(RandGauss()))));
  }

  std::vector<EventKeyType> special_keys;
  for (size_t i = 0; i < keys.size(); i++)
    if (RandUniform() < 0.5) special_keys.push_back(keys[i]);


  int32 nleaves = 0;
  EventMap *trivial_map = TrivialTree(&nleaves);


  // We do a complete split on these keys.
  EventMap *table_split_map = DoTableSplitMultiple(*trivial_map, special_keys,
                                                   stats, &nleaves);
  int32 nleaves_after_table_split = nleaves;
  std::cout << "TestClusterEventMapRestricted: after splitting on "<<special_keys.size()<<" keys, nleaves = " <<nleaves<<'\n';
  // We now do decision tree split.

  Questions qo;  // all default.
  int32 num_quest = rand() % 10, num_iters = rand () % 5;
  qo.InitRand(stats, num_quest, num_iters, kAllKeysInsistIdentical);
  float thresh = 0.001;
  int32 max_leaves = 50;
  BaseFloat smallest_split = 0.0;
  BaseFloat impr;
  EventMap *split_tree = SplitDecisionTree(*table_split_map, stats, qo, thresh, max_leaves,
                                           &nleaves, &impr, &smallest_split);
  KALDI_ASSERT((nleaves <= max_leaves || nleaves == nleaves_after_table_split) && smallest_split >= thresh);

  std::cout << "TestClusterEventMapRestricted: after building decision tree, " <<nleaves<<'\n';

  thresh = 1000;  // will ensure everything is combined.
  {
    int32 num_removed;
    EventMap *map_clustered = ClusterEventMap(*split_tree, stats,
                                              thresh, &num_removed);
    std::cout << "ClusterEventMap: num_removed = "<<num_removed;
    KALDI_ASSERT(num_removed == nleaves - 1);
    delete map_clustered;
  }

  {
    int32 num_removed;

    EventMap *map_clustered = NULL;
    if (test_by_key)
      map_clustered = ClusterEventMapRestrictedByKeys(*split_tree, stats,
                                                      thresh, special_keys,
                                                      &num_removed);
    else
      map_clustered = ClusterEventMapRestrictedByMap(*split_tree, stats,
                                                     thresh, *table_split_map,
                                                     &num_removed);

    std::cout << "ClusterEventMapRestricted: num_removed = "<<num_removed;
    // should take it back to status after table split.
    KALDI_ASSERT(num_removed == nleaves - nleaves_after_table_split);
    delete map_clustered;
  }

  delete split_tree;
  delete trivial_map;
  delete table_split_map;
  DeleteBuildTreeStats(&stats);
}


void TestShareEventMapLeaves() {
  // this is modified from TestClusterEventMapRestricted() [rather arbitrarily].


  int32 num_keys = 1 + rand() % 4;
  std::vector<EventKeyType> keys;

  {  // randomly choose keys.  Will always define all of them.
    std::set<EventKeyType> keys_set;
    while (keys_set.size() < (size_t)num_keys)
      keys_set.insert(  (rand() % (num_keys + 10)) - 3 );
    CopySetToVector(keys_set, &keys);
  }


  BuildTreeStatsType stats;

  int32 n_stats = 1 + (rand() % 10);
  n_stats *= n_stats;  // up to 81 stats.

  for (size_t i = 0; i < (size_t)n_stats; i++) {
    EventType evec;

    for (size_t j = 0; j < keys.size(); j++) {
      EventValueType val = rand() % 100;
      evec.push_back(std::make_pair(keys[j], val));
    }
    stats.push_back(std::make_pair(evec, static_cast<Clusterable*>(new ScalarClusterable(RandGauss()))));
  }

  std::vector<EventKeyType> special_keys;
  for (size_t i = 0; i < keys.size(); i++)
    if (RandUniform() < 0.5) special_keys.push_back(keys[i]);


  int32 nleaves = 0;
  EventMap *trivial_map = TrivialTree(&nleaves);


  // We do a complete split on these keys.
  EventMap *table_split_map = DoTableSplitMultiple(*trivial_map, special_keys,
                                                   stats, &nleaves);

  std::cout << "TestClusterEventMapRestricted: after splitting on "<<special_keys.size()<<" keys, nleaves = " <<nleaves<<'\n';
  // We now do decision tree split.
  int nleaves_after_table_split = nleaves;
  Questions qo;  // all default.
  int32 num_quest = rand() % 10, num_iters = rand () % 5;
  qo.InitRand(stats, num_quest, num_iters, kAllKeysInsistIdentical);
  float thresh = 0.001;
  int32 max_leaves = 100;
  BaseFloat impr;
  BaseFloat smallest_split;
  EventMap *split_tree = SplitDecisionTree(*table_split_map, stats, qo, thresh, max_leaves,
                                           &nleaves, &impr, &smallest_split);
  KALDI_ASSERT((nleaves <= max_leaves || nleaves == nleaves_after_table_split) && smallest_split >= thresh);

  std::cout << "TestShareEventMapLeaves: after building decision tree, " <<nleaves<<'\n';

  if (special_keys.size() == 0) {
    KALDI_WARN << "TestShareEventMapLeaves(): could not test since key not always defined.\n";
    delete split_tree;
    delete trivial_map;
    delete table_split_map;
    DeleteBuildTreeStats(&stats);
    return;
  }
  EventKeyType key = special_keys[rand() % special_keys.size()];
  std::vector<EventValueType> values;
  bool always_defined = PossibleValues(key, stats, &values);
  KALDI_ASSERT(always_defined);

  std::set<EventValueType> to_share;
  for (size_t i = 0; i < 3; i++) to_share.insert(values[rand() % values.size()]);

  std::vector<std::vector<EventValueType> > share_value;
  for (std::set<EventValueType>::iterator iter = to_share.begin();
      iter != to_share.end();
      iter++) {
    share_value.resize(share_value.size()+1);
    share_value.back().push_back(*iter);
  }
  int num_leaves;
  EventMap *shared = ShareEventMapLeaves(*split_tree,
                                         key,
                                         share_value,
                                         &num_leaves);
  KALDI_ASSERT(num_leaves <= nleaves);
  for (size_t i = 0; i < share_value.size(); i++) {
    EventType evec;
    std::vector<EventAnswerType> answers;
    evec.push_back(MakeEventPair(key, share_value[i][0]));
    shared->MultiMap(evec, &answers);
    SortAndUniq(&answers);
    KALDI_ASSERT(answers.size() == 1);  // should have been shared.
  }
  delete shared;

  delete split_tree;
  delete trivial_map;
  delete table_split_map;
  DeleteBuildTreeStats(&stats);
}


void TestQuestionsInitRand() {
  // testing Questions::InitRand() function.  Also testing I/O of Questions class.
  for (int32 p = 0;p < 10;p++) {
    std::vector<EventKeyType>  keys_all, keys_some;
    {
      std::set<EventKeyType> keys_all_set, keys_some_set;
      int32 num_all = rand() % 3, num_some = rand() % 3;
      for (int32 i = 0;i < num_all;i++) keys_all_set.insert(rand() % 10);
      for (int32 i = 0;i < num_some;i++) {
        int32 k = rand() % 10;
        if (keys_all_set.count(k) == 0) keys_some_set.insert(k);
      }
      CopySetToVector(keys_all_set, &keys_all);
      CopySetToVector(keys_some_set, &keys_some);
    }
    std::set<EventKeyType> keys_all_saw_set;
    // Now we have two distinct sets of keys keys_all and keys_some.
    // We now create the Clusterable* stuff.
    BuildTreeStatsType dummy_stats;  // dummy because the Clusterable *pointers are actually NULL.
    size_t n_stats = rand() % 100;
    // make sure we sometimes have empty or just one stats: may find extra bugs.
    if (n_stats > 90) n_stats = 0;
    if (n_stats > 80) n_stats = 1;

    for (size_t i = 0;i < n_stats;i++) {  // Create stats...
      EventType evec;
      for (size_t j = 0;j < keys_all.size();j++) {
        evec.push_back(std::make_pair( keys_all[j], (EventValueType)(rand() % 10)));
        keys_all_saw_set.insert(keys_all[j]);
      }
      for (size_t j = 0;j < keys_some.size();j++)
        if (rand() % 2 == 0) {  // randomly w.p. 1/2
          evec.push_back(std::make_pair( keys_some[j], (EventValueType)(rand() % 10)));
          keys_all_saw_set.insert(keys_some[j]);
        }
      std::sort(evec.begin(), evec.end());  // sorts on keys.
      EventMap::Check(evec);
      dummy_stats.push_back(std::make_pair(evec, (Clusterable*)NULL));
    }
    Questions qo;  // all default.
    bool intersection = (p%2 == 0);
    int32 num_quest = rand() % 10, num_iters = rand () % 5;
    qo.InitRand(dummy_stats, num_quest, num_iters, intersection ? kAllKeysIntersection : kAllKeysUnion);

    for (int i = 0; i < 2; i++) {
      // Here, test I/O of questions class.
      bool binary = (i == 0);
      std::ostringstream oss;
      qo.Write(oss, binary);

      std::istringstream iss(oss.str());
      Questions qo2;
      qo2.Read(iss, binary);

      std::ostringstream oss2;
      qo2.Write(oss2, binary);

      if (oss.str() != oss2.str()) {
        KALDI_ERR << "Questions I/O failure: " << oss.str() << " vs. " << oss2.str();
      }
    }

    if (n_stats > 0) {
      if (p < 2) {
        for (size_t i = 0;i < keys_all.size();i++) {
          KALDI_ASSERT(qo.HasQuestionsForKey(keys_all[i]));
          const QuestionsForKey &opts = qo.GetQuestionsOf(keys_all[i]);
          std::cout << "num-quest: "<< opts.initial_questions.size() << '\n';
          for (size_t j = 0;j < opts.initial_questions.size();j++) {
            for (size_t k = 0;k < opts.initial_questions[j].size();k++)
              std::cout << opts.initial_questions[j][k] <<" ";
            std::cout << '\n';
          }
        }
      }
      if (intersection) {
        for (size_t i = 0;i < keys_all.size();i++) {
          KALDI_ASSERT(qo.HasQuestionsForKey(keys_all[i]));
          qo.GetQuestionsOf(keys_all[i]);
        }
      } else {  // union: expect to see all keys that were in the data.
        for (std::set<int32>::iterator iter = keys_all_saw_set.begin(); iter != keys_all_saw_set.end(); iter++) {
          KALDI_ASSERT(qo.HasQuestionsForKey(*iter));
        }
      }
    }
  }
}


void TestSplitDecisionTree() {
  // part of the code is the same as the code for testing Questions::InitRand() function.
  for (int32 p = 0;p < 4;p++) {
    std::vector<EventKeyType>  keys_all, keys_some;
    {
      std::set<EventKeyType> keys_all_set, keys_some_set;
      int32 num_all = rand() % 3, num_some = rand() % 3;
      for (int32 i = 0;i < num_all;i++) keys_all_set.insert(rand() % 10);
      for (int32 i = 0;i < num_some;i++) {
        int32 k = rand() % 10;
        if (keys_all_set.count(k) == 0) keys_some_set.insert(k);
      }
      CopySetToVector(keys_all_set, &keys_all);
      CopySetToVector(keys_some_set, &keys_some);
    }
    std::set<EventKeyType> keys_all_saw_set;
    // Now we have two distinct sets of keys keys_all and keys_some.
    // We now create the Clusterable* stuff.
    BuildTreeStatsType stats;  // dummy because the Clusterable *pointers are actually NULL.
    size_t n_stats = rand() % 100;
    // make sure we sometimes have empty or just one stats: may find extra bugs.
    if (n_stats > 90) n_stats = 0;
    if (n_stats > 80) n_stats = 1;

    for (size_t i = 0;i < n_stats;i++) {  // Create stats...
      EventType evec;
      for (size_t j = 0;j < keys_all.size();j++) {
        evec.push_back(std::make_pair( keys_all[j], (EventValueType)(rand() % 10)));
        keys_all_saw_set.insert(keys_all[j]);
      }
      for (size_t j = 0;j < keys_some.size();j++)
        if (rand() % 2 == 0) {  // randomly w.p. 1/2
          evec.push_back(std::make_pair( keys_some[j], (EventValueType)(rand() % 10)));
          keys_all_saw_set.insert(keys_some[j]);
        }
      std::sort(evec.begin(), evec.end());  // sorts on keys.
      EventMap::Check(evec);
      stats.push_back(std::make_pair(evec, (Clusterable*) new ScalarClusterable(RandGauss())));
    }
    Questions qo;  // all default.
    BaseFloat thresh = 0.00001;  // these stats have a count of 1... need v. small thresh to get them to merge.
    // 0.000001 tries to ensure everything is split.

    bool intersection = true;  // keep borrowed code later on happy.

    int32 num_quest = rand() % 10, num_iters = rand () % 5;
    qo.InitRand(stats, num_quest, num_iters, kAllKeysIntersection);

    if (n_stats > 0) {
      if (p < 2) {
        for (size_t i = 0;i < keys_all.size();i++) {
          KALDI_ASSERT(qo.HasQuestionsForKey(keys_all[i]));
          const QuestionsForKey &opts = qo.GetQuestionsOf(keys_all[i]);
          std::cout << "num-quest: "<< opts.initial_questions.size() << '\n';
          for (size_t j = 0;j < opts.initial_questions.size();j++) {
            for (size_t k = 0;k < opts.initial_questions[j].size();k++)
              std::cout << opts.initial_questions[j][k] <<" ";
            std::cout << '\n';
          }
        }
      }
      if (intersection) {
        for (size_t i = 0;i < keys_all.size();i++) {
          KALDI_ASSERT(qo.HasQuestionsForKey(keys_all[i]));
          qo.GetQuestionsOf(keys_all[i]);
        }
      } else {  // union: expect to see all keys that were in the data.
        for (std::set<int32>::iterator iter = keys_all_saw_set.begin(); iter != keys_all_saw_set.end(); iter++) {
          KALDI_ASSERT(qo.HasQuestionsForKey(*iter));
        }
      }
      std::cout << "num_quest = " <<num_quest<<", num_iters = "<<num_iters<<'\n';
      // OK, now want to do decision-tree split.

      int32 num_leaves = 0;
      int32 max_leaves = 50;
      EventMap *trivial_tree = TrivialTree(&num_leaves);

      BaseFloat impr, smallest_split;
      EventMap *split_tree = SplitDecisionTree(*trivial_tree, stats, qo, thresh, max_leaves,
                                               &num_leaves, &impr, &smallest_split);
      KALDI_ASSERT(num_leaves <= max_leaves && smallest_split >= thresh);

      {
        BaseFloat impr_check = ObjfGivenMap(stats, *split_tree) - ObjfGivenMap(stats, *trivial_tree);
        std::cout << "Objf impr is " << impr << ", computed differently: " <<impr_check<<'\n';
        KALDI_ASSERT(fabs(impr - impr_check) < 0.1);
      }


      std::cout << "After splitting, num_leaves = " << num_leaves << '\n';

      std::vector<BuildTreeStatsType> mapped_stats;
      SplitStatsByMap(stats, *split_tree, &mapped_stats);
      std::cout << "Assignments of stats to leaves is:\n";
      for (size_t i = 0; i < mapped_stats.size(); i++) {
        std::cout << " [ leaf "<<i<<"]: ";
        for (size_t j = 0; j < mapped_stats[i].size(); j++) {
          std::cout << ((ScalarClusterable*)(mapped_stats[i][j].second))->Info() << " ";
        }
        std::cout << '\n';
      }
      delete trivial_tree;
      delete split_tree;
      DeleteBuildTreeStats(&stats);
    }
  }
}
void TestBuildTreeStatsIo(bool binary) {
  for (int32 p = 0; p < 10; p++) {
    size_t num_stats = rand() % 20;
    BuildTreeStatsType stats;
    for (size_t i = 0; i < num_stats; i++) {
      EventType ev;
      int32 ev_sz = rand() % 5;
      for (int32 i = 0; i < ev_sz; i++) {
        EventKeyType key = (i == 0 ? 0 : ev[i-1].first) + rand() % 2, value = rand() % 10;
        ev.push_back(std::make_pair(key, value));
      }
      stats.push_back(std::make_pair(ev, (Clusterable*) NULL));
    }
    const char *filename = "tmpf";
    WriteBuildTreeStats(Output(filename, binary).Stream(), binary, stats);

    {
      bool binary_in;
      BuildTreeStatsType stats2;
      GaussClusterable gc;  // just need some random Clusterable object
      Input ki(filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(),
                         binary_in, gc, &stats2);
      KALDI_ASSERT(stats == stats2);
    }
  }
}



} // end namespace kaldi

int main() {
  using namespace kaldi;
  for (size_t i = 0;i < 2;i++) {
    TestTrivialTree();
    TestPossibleValues();
    TestDoTableSplit();
    TestClusterEventMapGetMappingAndRenumberEventMap();
    TestClusterEventMapGetMappingAndRenumberEventMap2();  // tests "with gaps"
    TestClusterEventMap();
    TestClusterEventMapRestricted();
    TestShareEventMapLeaves();
    TestQuestionsInitRand();
    TestSplitDecisionTree();
    TestBuildTreeStatsIo(false);
    TestBuildTreeStatsIo(true);
    TestConvertStats();
    TestSplitStatsByKey(); 
    TestFindAllKeys();  // put at end because it throws+catches internally.
  }
}

