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

struct PairIsEqualComparator  :
    public std::unary_function<std::pair<int32, int32>, bool>
{
  explicit PairIsEqualComparator(const std::pair<int32, int32> pair):
      pair_(pair) {}
  bool operator() (std::pair<int32, int32> const &arg)
  {
    if (pair_.first == arg.first)
      return pair_.second == arg.second;
    return false;
  }
  std::pair<int32, int32> pair_;
};

void PrintVectorVectorPair(
    std::vector<std::vector<std::pair<int32, int32> > > vec_vec_pair)  {
  std::ostringstream ostream;
  for (int32 i = 0; i < vec_vec_pair.size(); i++) {
    for (int32 j = 0; j < vec_vec_pair[i].size(); j++)  {
      ostream << "(" << vec_vec_pair[i][j].first << ","
              << vec_vec_pair[i][j].second << ") ";
    }
    ostream << std::endl;
  }
  KALDI_LOG << ostream.str();
}

// Function to check SplitLocationsBackward() method
// checks if the submat_lists and split_lists have the same non-dummy elements
// checks if the submat_lists are split into same first_element lists wherever
// possible
// checks if the split_lists satisfy either "unique contiguous segments"
// property or unique pairs property (see SplitLocationsBackward in
// nnet-compile-utils.h for more details)
void UnitTestSplitLocationsBackward(bool verbose) {
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

  SplitLocationsBackward(submat_lists, &split_lists);
  // Checking split_lists has all the necessary properties
  for (int32 i = 0; i < split_lists.size(); i++)  {
    int32 first_value;
    std::vector<int32> second_values;
    if (ConvertToIndexes(split_lists[i], &first_value, &second_values))  {
      // checking for contiguity and uniqueness of .second elements
      std::vector<int32> occurred_values;
      int32 prev_value = -10; // using a negative value as all indices are > 0
      for (int32 j = 0; j < second_values.size(); j++)  {
        if (second_values[j] == -1)
          continue;
        if (second_values[j] != prev_value) {
          std::vector<int32>::iterator iter = std::find(occurred_values.begin(),
                                                        occurred_values.end(),
                                                        second_values[j]);
          KALDI_ASSERT(iter == occurred_values.end());
        }
      }
    } else {
      std::vector<std::pair<int32, int32> > list_of_pairs;
      // checking for uniques of elements in the list
      for (int32 j = 0; j < split_lists[i].size(); j++)  {
        if (split_lists[i][j].first == -1)
          continue;
        std::vector<std::pair<int32, int32> >::const_iterator iter =
            std::find_if(list_of_pairs.begin(), list_of_pairs.end(),
                         PairIsEqualComparator(split_lists[i][j]));
        KALDI_ASSERT(iter == list_of_pairs.end());
        list_of_pairs.push_back(split_lists[i][j]);
      }
    }
  }
  if (verbose)  {
    KALDI_LOG << "submat_list";
    PrintVectorVectorPair(submat_lists);
    KALDI_LOG << "split_lists";
    PrintVectorVectorPair(split_lists);
    KALDI_LOG << "===========================";
  }
  int32 num_kAddRows_in_output = 0;
  int32 first_value;
  std::vector<int32> second_values;
  // ensure that elements in submat_lists are also present
  // in split_lists
  for (int32 i = 0 ; i < split_lists.size(); i++) {
    second_values.clear();
    if (ConvertToIndexes(split_lists[i], &first_value, &second_values)) {
      // Checking if ConvertToIndexes did a proper conversion of the indexes
      KALDI_ASSERT(second_values.size() == split_lists[i].size());
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


void UnitTestHasContiguousProperty() {
  for (int32 k = 0; k < 10; k++) {
    int32 size = RandInt(0, 5);
    std::vector<int32> indexes(size);
    for (int32 i = 0; i < size; i++)
      indexes[i] = RandInt(-1, 4);
    std::vector<std::pair<int32, int32> > reverse_indexes;
    bool ans = HasContiguousProperty(indexes, &reverse_indexes);
    if (!ans) { // doesn't have contiguous propety.
      KALDI_LOG << "no.";
      bool found_example = false;
      for (int32 i = 0; i < size; i++) {
        if (indexes[i] != -1) {
          bool found_not_same = false;
          for (int32 j = i + 1; j < size; j++) {
            if (indexes[j] != indexes[i]) found_not_same = true;
            else if (found_not_same) found_example = true;  // found something like x y x.
          }
        }
      }
      KALDI_ASSERT(found_example);
    } else {
      KALDI_LOG << "yes.";
      for (int32 i = 0; i < reverse_indexes.size(); i++) {
        for (int32 j = reverse_indexes[i].first;
             j < reverse_indexes[i].second; j++) {
          KALDI_ASSERT(indexes[j] == i);
          indexes[j] = -1;
        }
      }
      for (int32 i = 0; i < size; i++)  // make sure all indexes covered.
        KALDI_ASSERT(indexes[i] == -1);
    }
  }
}


void UnitTestEnsureContiguousProperty() {
  for (int32 k = 0; k < 10; k++) {
    int32 size = RandInt(0, 5);
    std::vector<int32> indexes(size);
    for (int32 i = 0; i < size; i++)
      indexes[i] = RandInt(-1, 4);
    std::vector<std::pair<int32, int32> > reverse_indexes;
    bool ans = HasContiguousProperty(indexes, &reverse_indexes);
    if (ans) { // has contiguous property -> EnsureContiguousProperty should do
               // nothing.
      std::vector<std::vector<int32> > indexes_split;
      EnsureContiguousProperty(indexes, &indexes_split);
      if (indexes.size() == 0 ||
          *std::max_element(indexes.begin(), indexes.end()) == -1) {
        KALDI_ASSERT(indexes_split.size() == 0);
      } else {
        KALDI_ASSERT(indexes_split.size() == 1 &&
                     indexes_split[0] == indexes);
      }
    } else {
      std::vector<std::vector<int32> > indexes_split;
      EnsureContiguousProperty(indexes, &indexes_split);
      KALDI_ASSERT(indexes_split.size() > 1);
      for (int32 i = 0; i < indexes.size(); i++) {
        int32 this_val = indexes[i];
        bool found = (this_val == -1);  // not looking for anything if
                                        // this_val is -1.
        for (int32 j = 0; j < indexes_split.size(); j++) {
          if (found) {
            KALDI_ASSERT(indexes_split[j][i] == -1);
          } else {
            if (indexes_split[j][i] == this_val) {
              found = true;
            } else {
              KALDI_ASSERT(indexes_split[j][i] == -1);
            }
          }
        }
        KALDI_ASSERT(found);
        for (int32 j = 0; j < indexes_split.size(); j++) {
          KALDI_ASSERT(indexes_split[j].size() == indexes.size() &&
                       HasContiguousProperty(indexes_split[j], &reverse_indexes));
        }
      }
    }
  }
}


// Function to check SplitLocations() method
// checks if the submat_lists and split_lists have the same non-dummy elements
// checks if the submat_lists are split into same first_element lists wherever
// possible
void UnitTestSplitLocations(bool verbose) {
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
  if (verbose)  {
    KALDI_LOG << "submat_list";
    PrintVectorVectorPair(submat_lists);
    KALDI_LOG << "split_lists";
    PrintVectorVectorPair(split_lists);
    KALDI_LOG << "===========================";
    KALDI_LOG << split_lists.size();
  }
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
  bool verbose = false;
  for (int32 loop = 0; loop < 10; loop++)  {
    UnitTestSplitLocations(verbose);
    UnitTestSplitLocationsBackward(verbose);
    UnitTestHasContiguousProperty();
    UnitTestEnsureContiguousProperty();
  }
  KALDI_LOG << "Tests passed.";
  return 0;
}
