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

#include "util/common-utils.h"
#include "nnet3/nnet-compile-utils.h"

namespace kaldi {
namespace nnet3 {

struct ComparePair : public std::unary_function<std::pair<int32, int32>, bool>
{
  explicit ComparePair(const std::pair<int32, int32> &correct_pair):
  correct_pair_(correct_pair) {}
  bool operator() (std::pair<int32, int32> const &arg)
  { return (arg.first == correct_pair_.first) &&
           (arg.second == correct_pair_.second); }
  std::pair<int32, int32> correct_pair_;
};

// Function to check SplitLocations() method
// checks if the submat_lists and split_lists have the same non-dummy elements
// checks if the submat_lists are split into same first_element lists wherever
// possible
void UnitTestSplitLocations() {
  int32 minibatch_size = Rand() % 1024 + 100;
  int32 num_submat_indexes = Rand() % 10 + 1;
  int32 max_submat_list_size = Rand() % 10 + 1;
  int32 min_num_kAddRows = Rand() % 2; // minimum number of kAddRows compatible
  // lists expected in the final split lists. This value will be used to
  // create input submat_lists so that this is guaranteed
  max_submat_list_size = min_num_kAddRows + max_submat_list_size;

  std::vector<std::pair<int32, int32> > all_pairs;
  all_pairs.reserve(minibatch_size * max_submat_list_size);
  std::vector<std::vector<std::pair<int32, int32> > > 
      submat_lists(minibatch_size),
      split_lists;
  std::vector<int32> submat_indexes(num_submat_indexes);
  for (int32 i = 0; i < num_submat_indexes; i++)  {
    submat_indexes[i] = Rand();
  }

  // generating submat_lists
  int32 max_generated_submat_list_size = 0;
  for (int32 i = 0; i < minibatch_size; i++)  {
    int32 num_locations = Rand() % max_submat_list_size + 1;
    max_generated_submat_list_size =
        max_generated_submat_list_size < num_locations ?
        num_locations : max_generated_submat_list_size;
    submat_lists[i].reserve(num_locations);
    for (int32 j = 0; j < num_locations; j++) {
      if (j <= min_num_kAddRows)
        // since we need min_num_kAddRows in the split_lists we ensure that
        // we add a pair with the same first element in all the submat_lists 
        submat_lists[i].push_back(std::make_pair(submat_indexes[j],
                           Rand() % minibatch_size));
        
      submat_lists[i].push_back(
          std::make_pair(submat_indexes[Rand() % num_submat_indexes],
                         Rand() % minibatch_size));
    }
    all_pairs.insert(all_pairs.end(), submat_lists[i].begin(),
                     submat_lists[i].end());
  }
  SplitLocations(submat_lists, &split_lists);

  int32 num_kAddRows_in_output = 0;
  int32 first_value;
  std::vector<int32> second_values;
  // ensure that elements in submat_lists are also present
  // in split_lists
  for (int32 i = 0 ; i < split_lists.size(); i++) {
    second_values.clear();
    if (ConvertToIndexes(split_lists[i], &first_value, &second_values)) {
      // Checking if ConvertToIndexes did a proper conversion of the indexes
      for (int32 j = 0; j < second_values.size(); j++)  {
        if (split_lists[i][j].first != -1)
          KALDI_ASSERT((split_lists[i][j].first == first_value) &&
                       (split_lists[i][j].second == second_values[j]));
      }
      num_kAddRows_in_output++;
    }
    for (int32 j = 0; j < split_lists[i].size(); j++) {
      if (split_lists[i][j].first == -1)
        continue;
      std::vector<std::pair<int32, int32> >::iterator iter =
          std::find_if(all_pairs.begin(), all_pairs.end(),
                    ComparePair(split_lists[i][j]));
      KALDI_ASSERT(iter != all_pairs.end());
      all_pairs.erase(iter);
    }
  }
  KALDI_ASSERT(all_pairs.size() == 0);
  // ensure that there are at least as many kAddRows compatible split_lists as
  // specified
  KALDI_ASSERT(num_kAddRows_in_output >= min_num_kAddRows);
}

} // namespace nnet2
} // namespace kaldi

int main()  {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  for (int32 loop = 0; loop < 10; loop++)  {
    UnitTestSplitLocations();
  }
  KALDI_LOG << "Tests passed.";
  return 0;
}
