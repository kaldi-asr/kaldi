// tree/event-map-test.cc

// Copyright 2009-2011  Microsoft Corporation  Haihua Xu  Yanmin Qian

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
  ConstantEventMap *C1b = new ConstantEventMap(1);
  std::vector<EventMap*> tvec;
  tvec.push_back(C0a);
  tvec.push_back(C1b);

  TableEventMap *T1 = new TableEventMap(1, tvec);  // takes ownership of C0a, C1b
  assert(T1->MaxResult() == 1);

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
    size_t nElems = rand() % 10;  // num of value->answer pairs.
    std::map<ValueType, AnswerType> init_map;
    for (size_t i = 0;i < nElems;i++) {
      init_map[rand() % 10] = rand() % 5;
    }
    EventKeyType key = rand() % 10;
    TableEventMap T3(key, init_map);
    for (size_t i = 0; i < 10; i++) {
      EventType vec;
      vec.push_back(std::make_pair(key, (ValueType) i));
      AnswerType ans;
      // T3.Map(vec, &ans);
      if (init_map.count(i) == 0) {
        assert( ! T3.Map(vec, &ans) );  // false
      } else {
        bool b = T3.Map(vec, &ans);
        assert(b);
        assert(ans == init_map[i]);  // true
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
    size_t size = rand() % 20;
    event_vec.resize(size);
    for (size_t i = 0;i < size;i++) {
      event_vec[i].first = rand() % 10 + (i > 0 ? event_vec[i-1].first : 0);
      event_vec[i].second = rand() % 20;
    }


    {
      const char *filename = "tmpf";
      Output ko(filename, binary);
      std::ostream &outfile = ko.Stream();
      WriteEventType(outfile, binary, event_vec);
      ko.Close();

      {
        bool binary_in;
        Input is(filename, &binary_in);
        std::istream &infile = is.Stream();
        EventType evec2;
        evec2.push_back(std::make_pair<EventKeyType, EventValueType>(1, 1));  // make it nonempty.
        ReadEventType(infile, binary_in, &evec2);
        assert(evec2 == event_vec);
      }
    }
  }
}

EventMap *RandomEventMap(const std::vector<EventKeyType> &keys) {
  // Do not mess with the probabilities inside this routine or there
  // is a danger this function will blow up.
  int32 max_val = 20;
  assert(keys.size() != 0);
  float f = RandUniform();
  if (f < 0.333) {  // w.p. 0.333, return ConstantEventMap.
    return new ConstantEventMap(rand() % max_val);
  } else if (f < 0.666) {  // w.p. 0.333, return TableEventMap.
    float nonnull_prob = 0.3;  // prob of a non-NULL pointer.
    float expected_table_size = 3.0;
    int32 table_size = RandPoisson(expected_table_size);
    // fertility from this branch is 0.333 * 3.0 * 0.2333 = 0.3.
    EventKeyType key = keys[rand() % keys.size()];
    std::vector<EventMap*> table(table_size);
    for (size_t t = 0; t < (size_t)table_size; t++) {
      if (RandUniform() < nonnull_prob) table[t] = RandomEventMap(keys);
      else table[t] = NULL;
    }
    return new TableEventMap(key, table);
  } else {  // w.p. 0.333, return SplitEventMap.
    // Fertility of this stage is 0.333 * 2 = 0.666.
    EventKeyType key = keys[rand() % keys.size()];
    std::set<EventValueType> yes_set;
    for (size_t i = 0; i < 5; i++) yes_set.insert(rand() % max_val);
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
    int32 num_keys = 1 + (rand() % (max_key - 1));
    std::set<EventKeyType> key_set;
    // - 5 to allow negative keys.  These are allowed.
    while (key_set.size() < (size_t)num_keys) key_set.insert( (rand() % (2*max_key)) - 5);
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
    assert(str_out.str() == str2_out.str());
    delete read_map;
    delete rand_map;
  }
}



} // end namespace kaldi




int main() {
  kaldi::TestEventMap();
  kaldi::TestEventTypeIo(false);
  kaldi::TestEventTypeIo(true);
  kaldi::TestEventMapIo(false);
  kaldi::TestEventMapIo(true);
}
