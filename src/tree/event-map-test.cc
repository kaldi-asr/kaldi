// tree/event-map-test.cc

// Copyright 2009-2011  Microsoft Corporation;  Haihua Xu;  Yanmin Qian
//                2013  Johns Hopkins University (author: Daniel Povey)

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

#include "tree/event-map.h"
#include "util/kaldi-io.h"
#include <map>

namespace kaldi {

void TestEventMap() {
  typedef EventKeyType KeyType;
  typedef EventValueType ValueType;
  typedef EventAnswerType AnswerType;


  ConstantEventMap *C0a = new ConstantEventMap(0);
  {
    int32 num_leaves;
    std::vector<int32> parents;
    bool a = GetTreeStructure(*C0a, &num_leaves, &parents);
    KALDI_ASSERT(a && parents.size() == 1 && parents[0] == 0);
  }
  ConstantEventMap *C1b = new ConstantEventMap(1);
  {
    int32 num_leaves;
    std::vector<int32> parents;
    bool a = GetTreeStructure(*C1b, &num_leaves, &parents);
    KALDI_ASSERT(!a); // since C1b's leaves don't start from 0.
  }
  
  std::vector<EventMap*> tvec;
  tvec.push_back(C0a);
  tvec.push_back(C1b);

  TableEventMap *T1 = new TableEventMap(1, tvec);  // takes ownership of C0a, C1b
  KALDI_ASSERT(T1->MaxResult() == 1);

  {
    int32 num_leaves;
    std::vector<int32> parents;
    bool a = GetTreeStructure(*T1, &num_leaves, &parents);
    KALDI_ASSERT(a && parents.size() == 3 && parents[0] == 2
                 && parents[1] == 2 && parents[2] == 2);
  }
  
  ConstantEventMap *C0c = new ConstantEventMap(0);
  ConstantEventMap *C1d = new ConstantEventMap(1);

  std::map<ValueType, EventMap*> tmap;
  tmap[0] = C0c; tmap[1] = C1d;
  TableEventMap *T2 = new TableEventMap(1, tmap);  // takes ownership of pointers C0c and C1d.

  std::vector<ValueType> vec;
  vec.push_back(4);
  vec.push_back(5);


  ConstantEventMap *D1 = new ConstantEventMap(10);  // owned by D3 below
  ConstantEventMap *D2 = new ConstantEventMap(15);  // owned by D3 below

  SplitEventMap *D3 = new   SplitEventMap(1, vec, D1, D2);

  // Test different initializer  for TableEventMap where input maps ints to ints.
  for (size_t i = 0;i < 100;i++) {
    size_t nElems = Rand() % 10;  // num of value->answer pairs.
    std::map<ValueType, AnswerType> init_map;
    for (size_t i = 0;i < nElems;i++) {
      init_map[Rand() % 10] = Rand() % 5;
    }
    EventKeyType key = Rand() % 10;
    TableEventMap T3(key, init_map);
    for (size_t i = 0; i < 10; i++) {
      EventType vec;
      vec.push_back(std::make_pair(key, (ValueType) i));
      AnswerType ans;
      // T3.Map(vec, &ans);
      if (init_map.count(i) == 0) {
        KALDI_ASSERT( ! T3.Map(vec, &ans) );  // false
      } else {
        bool b = T3.Map(vec, &ans);
        KALDI_ASSERT(b);
        KALDI_ASSERT(ans == init_map[i]);  // true
      }
    }
  }

  delete T1;
  delete T2;
  delete D3;
}


void TestEventTypeIo(bool binary) {
  for (size_t p = 0; p < 20; p++) {
    EventType event_vec;
    size_t size = Rand() % 20;
    event_vec.resize(size);
    for (size_t i = 0;i < size;i++) {
      event_vec[i].first = Rand() % 10 + (i > 0 ? event_vec[i-1].first : 0);
      event_vec[i].second = Rand() % 20;
    }


    {
      const char *filename = "tmpf";
      Output ko(filename, binary);
      std::ostream &outfile = ko.Stream();
      WriteEventType(outfile, binary, event_vec);
      ko.Close();

      {
        bool binary_in;
        Input ki(filename, &binary_in);
        std::istream &infile = ki.Stream();
        EventType evec2;
        evec2.push_back(std::pair<EventKeyType, EventValueType>(1, 1));  // make it nonempty.
        ReadEventType(infile, binary_in, &evec2);
        KALDI_ASSERT(evec2 == event_vec);
      }
    }
  }

  unlink("tmpf");
}

const int32 kMaxVal = 20;

EventMap *RandomEventMap(const std::vector<EventKeyType> &keys) {
  // Do not mess with the probabilities inside this routine or there
  // is a danger this function will blow up.
  int32 max_val = kMaxVal;
  KALDI_ASSERT(keys.size() != 0);
  float f = RandUniform();
  if (f < 0.333) {  // w.p. 0.333, return ConstantEventMap.
    return new ConstantEventMap(Rand() % max_val);
  } else if (f < 0.666) {  // w.p. 0.333, return TableEventMap.
    float nonnull_prob = 0.3;  // prob of a non-NULL pointer.
    float expected_table_size = 3.0;
    int32 table_size = RandPoisson(expected_table_size);
    // fertility from this branch is 0.333 * 3.0 * 0.2333 = 0.3.
    EventKeyType key = keys[Rand() % keys.size()];
    std::vector<EventMap*> table(table_size);
    for (size_t t = 0; t < (size_t)table_size; t++) {
      if (RandUniform() < nonnull_prob) table[t] = RandomEventMap(keys);
      else table[t] = NULL;
    }
    return new TableEventMap(key, table);
  } else {  // w.p. 0.333, return SplitEventMap.
    // Fertility of this stage is 0.333 * 2 = 0.666.
    EventKeyType key = keys[Rand() % keys.size()];
    std::set<EventValueType> yes_set;
    for (size_t i = 0; i < 5; i++) yes_set.insert(Rand() % max_val);
    std::vector<EventValueType> yes_vec;
    CopySetToVector(yes_set, &yes_vec);
    EventMap *yes = RandomEventMap(keys), *no = RandomEventMap(keys);
    return new SplitEventMap(key, yes_vec, yes, no);
  }
  // total fertility is 0.3 + 0.666 = 0.9666, hence this will terminate with finite memory (w.p.1)
}

void TestEventMapIo(bool binary) {
  for (size_t p = 0; p < 20; p++) {
    int32 max_key = 10;
    int32 num_keys = 1 + (Rand() % (max_key - 1));
    std::set<EventKeyType> key_set;
    // - 5 to allow negative keys.  These are allowed.
    while (key_set.size() < (size_t)num_keys) key_set.insert( (Rand() % (2*max_key)) - 5);
    std::vector<EventKeyType> key_vec;
    CopySetToVector(key_set, &key_vec);
    EventMap *rand_map = RandomEventMap(key_vec);

    std::ostringstream str_out;
    EventMap::Write(str_out, binary, rand_map);


    if (p < 1) {
      std::cout << "Random map is: "<<str_out.str()<<'\n';
    }

    std::istringstream str_in(str_out.str());

    EventMap *read_map = EventMap::Read(str_in, binary);
    std::ostringstream str2_out;
    EventMap::Write(str2_out, binary, read_map);

    // Checking we can write the map, read it in, and get the same string form.
    KALDI_ASSERT(str_out.str() == str2_out.str());
    delete read_map;
    delete rand_map;
  }
}

void TestEventMapPrune() {
  const EventAnswerType no_ans = -10;
  std::vector<EventKeyType> keys;
  keys.push_back(1); // these keys are 
  keys.push_back(2); // hardwired into the code below, do not change
  EventMap *em = RandomEventMap(keys);
  EventType empty_event;
  std::vector<EventAnswerType> all_answers;
  em->MultiMap(empty_event, &all_answers);
  SortAndUniq(&all_answers);
  std::vector<EventMap*> new_leaves;
  std::vector<EventAnswerType> mapping;
  for (size_t i = 0; i < all_answers.size(); i++) {
    EventAnswerType ans = all_answers[i];
    KALDI_ASSERT(ans >= 0);
    new_leaves.resize(ans + 1, NULL);
    mapping.resize(ans + 1, no_ans);
    EventAnswerType map_to;
    if (Rand() % 2 == 0) map_to = -1;
    else map_to = Rand() % 20;
    new_leaves[ans] = new ConstantEventMap(map_to);
    mapping[ans] = map_to;
  }
  EventMap *mapped_em = em->Copy(new_leaves),
      *pruned_em = mapped_em->Prune();
  for (size_t i = 0; i < new_leaves.size(); i++)
    delete new_leaves[i];
  for (int32 i = 0; i < 10; i++) {
    EventType event;
    for (int32 key = 1; key <= 2; key++) {
      if (Rand() % 2 == 0) {
        EventValueType value = Rand() % 20;
        event.push_back(std::make_pair(key, value));
      }
    }
    EventAnswerType answer, answer2;
    if (em->Map(event, &answer)) {
      bool ret;
      if (pruned_em == NULL) ret = false;
      else ret = pruned_em->Map(event, &answer2);
      KALDI_ASSERT(answer >= 0);
      EventAnswerType mapped_ans = mapping[answer];
      KALDI_ASSERT(mapped_ans != no_ans);
      if (mapped_ans == -1) {
        if (ret == false)
          KALDI_LOG << "Answer was correctly pruned away.";
        else
          KALDI_LOG << "Answer was not pruned away [but this is not required]";
      } else {
        KALDI_ASSERT(ret == true);
        KALDI_ASSERT(answer2 == mapped_ans);
        KALDI_LOG << "Answers match " << answer << " -> " << answer2;
      }
    }
  }
  delete em;
  delete mapped_em;
  delete pruned_em;
}

void TestEventMapMapValues() {
  std::vector<EventKeyType> keys;
  keys.push_back(1); // these keys are 
  keys.push_back(2); // hardwired into the code below, do not change
  EventMap *em = RandomEventMap(keys);
  EventType empty_event;

  unordered_set<EventKeyType> mapped_keys;
  unordered_map<EventKeyType,EventKeyType> value_map;
  if (Rand() % 2 == 0) mapped_keys.insert(1);
  if (Rand() % 2 == 0) mapped_keys.insert(2);

  EventValueType v_offset = Rand() % kMaxVal;
  for (EventValueType v = 0; v < kMaxVal; v++)
    value_map[v] = (v + v_offset) % kMaxVal;
    
  EventMap *mapped_em = em->MapValues(mapped_keys, value_map);
  
  for (int32 i = 0; i < 10; i++) {
    EventType event, mapped_event;
    for (int32 key = 1; key <= 2; key++) {
      if (Rand() % 2 == 0) {
        EventValueType value = Rand() % kMaxVal;
        event.push_back(std::make_pair(key, value));
        EventValueType mapped_value;
        if (mapped_keys.count(key) == 0) mapped_value = value;
        else mapped_value = value_map[value];
        mapped_event.push_back(std::make_pair(key, mapped_value));
      }
    }
    EventAnswerType answer, answer2;
    if (em->Map(event, &answer)) {
      bool ret = mapped_em->Map(mapped_event, &answer2);
      KALDI_ASSERT(ret);
      KALDI_ASSERT(answer == answer2);
    }
  }
  delete em;
  delete mapped_em;
}



} // end namespace kaldi




int main() {
  using namespace kaldi;
  TestEventTypeIo(false);
  TestEventTypeIo(true);
  TestEventMapIo(false);
  TestEventMapIo(true);
  for (int32 i = 0; i <  10; i++) {
    TestEventMap();
    TestEventMapPrune();
    TestEventMapMapValues();
  }
}
