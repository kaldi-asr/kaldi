// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include <iterator>
#include <sstream>
#include "nnet3/nnet-compile-utils.h"

namespace kaldi {
namespace nnet3 {

bool pair_comparator(const std::pair<int32, int32>& first_pair, const std::pair<int32, int32>& second_pair) {
    return first_pair.second > second_pair.second;
}


void SplitLocations(
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    std::vector<std::vector<std::pair<int32, int32> > > *split_lists) {
  
  // (TODO : using index access of vectors for readability, check if this is a
  // performance bottleneck and convert to iterator access if necessary)
  
  // computing a histogram of the submat indexes in the submat_lists
  // each occurence in a given list is considered unique so we maintain
  // a vector to count each occurence seperately
  unordered_map<int32, std::vector<int32> > submat_histogram;
  int32 max_submat_list_size = 0;  // max size of a submat_list 

  for (int32 i = 0; i < submat_lists.size(); i++) {
    if (submat_lists[i].size() > max_submat_list_size)
      max_submat_list_size = submat_lists[i].size();
    // compute a histogram of occurences of a submat in each list
    unordered_map<int32, int32> submat_histogram_for_list; 
    for (int32 j = 0; j < submat_lists[i].size(); j++)  {
      total_elements += 1;
      std::unordered_map<int32, int32>::const_iterator iter1 =
        submat_occurences_for_list.find(submat_lists[i][j].first);
      if (iter1 != submat_histogram_for_list.end())  {
        iter1->second += 1;  
      } else  {
        submat_histogram_for_list[submat_lists[i][j].first] = 1;
      }
    }
    // update the submat_histogram with the counts from 
    // submat_histogram_for_list
    std::unordered_map<int32, int32>::iterator iter2;
    for (iter2 = submat_histogram_for_list.begin(),
         iter2 != submat_histogram_for_list.end(),
         ++iter2) {
      // get the count from the submat_histogram_for_list
      int32 count_in_list = iter2->second;
      // get the count_vector from the submat_histogram
      std::vector<int32> & count_vector = submat_histogram[iter2->first];
      // update the count of each occurence in the vector
      for (int32 x = 0; x < count_vector->size(); x++) {
        if (count_in_list > 0)  {
          count_vector->at(x) += 1;
          count_in_list--;
        } else  {
          break;
        }
      }
      // add additional counts for new occurrences
      while (count_in_list > 0) {
        count_vector.push_back(1);
        count_in_list--;
      }
    }
  }

  // copy the key, occurence_counts from submat_histogram to a vector
  std::vector<std::pair<int32, int32> > submat_histogram_vector;
  std::unordered_map<int32, std::vector<int32> >::iterator hist_iter;
  for (hist_iter = submat_histogram.begin();
       hist_iter != submat_histogram.end();
       ++hist_iter) {
    for (int32 i = 0; i < (hist_iter->second).size(); i++)  {
      submat_histogram_vector.push_backi(std::make_pair(hist_iter->first, (hist_iter->second)[i]))
    }
  }
  
  // sort the vector based on value
  std::sort(submat_histogram_vector.begin(), submat_histogram_vector.end(), pair_comparator);

  // determining which submat acceses can be collapsed into one list;
  // to determine cost of memory accesses we define some constants
  // for better optimization these have to be empirically tuned
  BaseFloat kAddRowsMulti_cost = 1;  
  BaseFloat kAddRows_cost = 0.5;  
  int32 num_kAddRowsMulti = max_submat_list_size;

  // TODO: write logic to decide kAddRows selection
  while ((num_kAddRowsMulti * kAddRowsMulti_cost) > (kAddRows_cost + (num_kAddRowsMulti - 1) * kAddRows_cost)) {
    // converting the vector of lists of pairs into a list of vectors of pairs
    for (int32 i = 0; i < submat_histogram_vector; i++) {
      submat_index = submat_histogram_vector[i].first;
      for (int32 j = 0; j < submat_lists.size(); j++) {
        
      }
    }
  }
}

bool ConvertToIndexes(
    const std::vector<std::pair<int32, int32> > &location_vector,
    int32 *first_value,    
    std::vector<int32> *second_values)  {
  //TODO 
}



} // namespace nnet3
} // namespace kaldi
