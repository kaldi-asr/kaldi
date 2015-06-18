// nnet3/nnet-compile-utils-test.cc

// Copyright 2015  Johns Hopkins University (author: Vijayaditya Peddinti)

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

#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include "nnet3/nnet-compile-utils.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet3 {

struct ComparePair : public std::unary_function<std::pair<int32, int32>, bool>
{
  explicit ComparePair(const std::pair<int32, int32> &correct_pair):
  correct_pair_(correct_pair) {}
  bool operator() (const std::pair<int32, int32> &arg)
  { return (arg.first == correct_pair.first) &&
           (arg.second == correct_pair.second); }
  std::pair<int32, int32> correct_pair;
}

void UnitTestSplitLocations() {
  // constructing a submat_list (see compile-utils.h for details)
  int32 minibatch_size = Rand() % 1024;
  int32 num_submat_indexes = Rand() % 10;
  int32 max_submat_list_size = Rand() % 10;

  std::vector<std::pair<int32, int32> > all_pairs;
  all_pairs.reserve(minibatch_size * max_submat_list_size);
  std::vector<std::vector<std::pair<int32, int32> > > submat_lists(
      minibatch_size), split_lists;
  std::vector<int32> submat_indexes(num_submat_indexes);
  for (int32 i = 0; i < num_submat_indexes; i++)  {
    submat_indexes[i] = Rand();
  }

  int32 max_generated_submat_list_size = 0;
  for (int32 i = 0; i < minibatch_size; i++)  {
    int32 num_locations = Rand() % max_submat_list_size;
    max_generated_submat_list_size =
        max_generated_submat_list_size < num_locations ?
        num_locations : max_generated_submat_list_size;
    submat_lists[i].reserve(num_locations);
    for (int32 j = 0; j < num_locations; j++) {
      submat_lists[i].push_back(
          std::make_pair(submat_indexes[Rand % num_submat_indexes],
                         Rand % minibatch_size));
    }
    all_pairs.insert(all_pairs.end(), submat_lists[i].begin(),
                     submat_lists[i].end());
  }
  SplitLocations(submat_lists, &split_lists);
  for (int32 i = 0 ; i < split_lists.size(); i++) {
    for (int32 j = 0; j < split_lists[i].size(); j++) {
      std::vector<std::pair<int32, int32> >::iterator iter =
          std::find(all_pairs.begin(), all_pairs.end(),
                    ComparePair(split_lists[i][j]));
      KALDI_ASSERT(iter != all_pairs.end());
      all_pairs.erase(iter);
    }
  }
  
}

int main()  {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  for (int32 loop = 0; loop < 2; loop++)  {
    UnitTestSplitLocations();
  }
  return 0;
}
