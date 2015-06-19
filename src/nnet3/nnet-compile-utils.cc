// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015                           (author: Vijayaditya Peddinti)

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
#include "util/common-utils.h"
#include "nnet3/nnet-compile-utils.h"

namespace kaldi {
namespace nnet3 {

// this comparator will be used to sort pairs using first_element
// we declare it as a struct as it will also be used by std::lower_bound
// method which will supply elements of different types to the function
struct FirstElementComparator {
  int first_element(int32 t) const {
    return t;
  }

  int first_element(std::pair<int32, int32> t) const  {
    return t.first;
  }

  template< typename T1, typename T2>
  bool operator()( T1 const & t1, T2 const & t2) const  {
    return first_element(t1) < first_element(t2);
  }
};

struct FirstElementIsEqualComparator :
    public std::unary_function<std::pair<int32, int32>, bool>
{
  explicit FirstElementIsEqualComparator(const int32 element):
  element_(element) {}
  bool operator() (std::pair<int32, int32> const &arg)
  { return (arg.first == element_); }
  int32 element_;
};



// this comparator will be used to sort pairs initially by second element in
// descending order and then by first element in descending order
bool SecondElementComparator(const std::pair<int32, int32>& first_pair,
                               const std::pair<int32, int32>& second_pair) {
    if (first_pair.second == second_pair.second)
      return first_pair.first > second_pair.first;
    return first_pair.second > second_pair.second;
}

// Function to compute a histogram of the submat_index,
// which is the first_element in the location pair. 
// The pairs are stored in vector of lists of pairs.
// The function also passes on some intermediate variables generated
// during computation to the caller function, as they can be used later
void ComputeSubmatIndexHistogram(
    // vector of list of location pairs
    const std::vector<std::vector<std::pair<int32, int32> > > submat_lists,
    // a copy of the input submat_lists where the lists are sorted 
    // (this will be used in the caller function for sort and find functions)
    std::vector<std::vector<std::pair<int32, int32> > > * sorted_submat_lists,
    // a histogram of submat_indexes where 
    // the keys are submat_indexes and values are a vector of frequencies
    // of first occurrence, second occurrence, etc. of a submat_index
    // in a submat_list
    unordered_map<int32, std::vector<int32> >* submat_histogram,
    // maximum size of the submat_lists 
    int32* max_submat_list_size
    ) {

  KALDI_ASSERT(submat_lists.size() > 0);
  // computing the submat_histogram
  for (int32 i = 0; i < submat_lists.size(); i++) {
    KALDI_ASSERT(submat_lists[i].size() > 0);
    if (submat_lists[i].size() > *max_submat_list_size)
      *max_submat_list_size = submat_lists[i].size();
    sorted_submat_lists->push_back(submat_lists[i]);
    std::sort(sorted_submat_lists->at(i).begin(),
              sorted_submat_lists->at(i).end(),
              FirstElementComparator());
    // counting the occurrences of each element in the current submat_list;
    // each new occurrence of the same element, in this list, is counted
    // as a seperate symbol for frequency counts
    int j = 0;
    unordered_map<int32, std::vector<int32> >::iterator histogram_iterator
        = submat_histogram->end();
    int32 repetition_count = 0;
    while (j < sorted_submat_lists->at(i).size()) {
      if ((histogram_iterator == submat_histogram->end()) ||
          (histogram_iterator->first != submat_lists[i][j].first)) {
        histogram_iterator = submat_histogram->find(submat_lists[i][j].first);
        repetition_count = 0;
        // if a histogram entry was not found for this submat_index, add one
        if (histogram_iterator == submat_histogram->end()) {
          (*submat_histogram)[submat_lists[i][j].first];
          histogram_iterator = submat_histogram->find(submat_lists[i][j].first);
        }
      }

      if (repetition_count >= (histogram_iterator->second).size()) {
        // this is the first time the submat_index repeated this many times
        // so add an entry for this in the count vector
        (histogram_iterator->second).push_back(1);
      } else {
        (histogram_iterator->second)[repetition_count]++;
      }
      repetition_count++;
      j++;
    }
  }
}


// Function to find the first occurrence of a submat_index in list of location 
// pairs from a vector of list of locations pairs.
// The occurrences are returned as a list of vector iterators,
// pointing to the position of the pair in the list or to the
// end of the list (when the pair is not present)
void FindSubmatIndexInSubmatLists(
    // pair to search for in the submat_lists
    int32 submat_index,
    // sorted_submat_lists is a pointer as we want non-const iterators in the
    // output 
    std::vector<std::vector<std::pair<int32, int32> > > *sorted_submat_lists,
    // a vector of iterators to store the location of the pairs
    std::vector<std::vector<std::pair<int32, int32> >::iterator>
      *output_iterator_list,
    // the max size of the submat_lists if the found pairs have been removed
    int32 *max_remaining_submat_list_size) {

  output_iterator_list->reserve(sorted_submat_lists->size());
  *max_remaining_submat_list_size = 0;
  for (int32 i = 0; i < sorted_submat_lists->size(); i++)  {
    std::vector< std::pair<int32, int32> > & submat_list =
        sorted_submat_lists->at(i);
    output_iterator_list->push_back(
        std::find_if(submat_list.begin(), submat_list.end(),
                     FirstElementIsEqualComparator(submat_index)));
    int32 remaining_submat_list_size = submat_list.size();
    if  (output_iterator_list->back() != submat_list.end())  {
      // since the submat_index is present in this submat_list
      // if submat_index was deleted from the list
      // the remaining submat_list's size is reduced by 1
      remaining_submat_list_size--;
    }
    *max_remaining_submat_list_size = remaining_submat_list_size
        > *max_remaining_submat_list_size ? remaining_submat_list_size :
        *max_remaining_submat_list_size;
  }
}

// Function to extract the identified pairs (using iterator)
// from a vector of list of pairs, to extract means to copy into
// a list and erase the original pair from the submat_lists
void ExtractGivenPairsFromSubmatLists(
    std::vector<std::vector<std::pair<int32, int32> >::iterator>
      &input_iterator_list,
    std::vector<std::vector<std::pair<int32, int32> > > *sorted_submat_lists,
    std::vector<std::pair<int32, int32> > *list_of_pairs) {
  list_of_pairs->reserve(sorted_submat_lists->size());
  for (int32 i = 0; i < input_iterator_list.size(); i++) {
    if (input_iterator_list[i] != sorted_submat_lists->at(i).end()) {
      // there was an element with the submat_index in the current list
      list_of_pairs->push_back(*input_iterator_list[i]);
      sorted_submat_lists->at(i).erase(input_iterator_list[i]);
    } else  {
      // insert a dummy element. Callers of this function expect the dummy
      // element to be (-1, -1)
      list_of_pairs->push_back(std::make_pair(-1, -1));
    }
  }
}

// Function to extract the last pairs from a vector of list of pairs
// a dummy is added when the list is empty
void ExtractLastPairFromSubmatLists(
    std::vector<std::vector<std::pair<int32, int32> > > *sorted_submat_lists,
    std::vector<std::pair<int32, int32> > *list_of_pairs) {
  list_of_pairs->reserve(sorted_submat_lists->size());
  for (int32 i = 0; i < sorted_submat_lists->size(); i++) {
    if (sorted_submat_lists->at(i).size() == 0)  {
      // the value of the dummy has to be (-1, -1) as down stream code has
      // expects -1 values for dummies
      list_of_pairs->push_back(std::make_pair(-1, -1));
      continue;
    }
    list_of_pairs->push_back(sorted_submat_lists->at(i).back());
    sorted_submat_lists->at(i).pop_back();
  }
}

// Function which does the actual splitting of submat_lists. But it operates on
// sorted submat_lists and uses submat_histogram_vector. 
// See SplitLocations, below for the algorithm
void SplitLocationsUsingSubmatHistogram(
    // maximum size of the lists in the sorted_submat_lists
    int32 max_submat_list_size,
    // a vector of list of pairs where each list is expected to be sorted
    // this is a pointer as the lists will be modified
    std::vector<std::vector<std::pair<int32, int32> > > *sorted_submat_lists,
    // a vector of pairs to represent a histogram
    // this is a pointer as the vector will be sorted
    std::vector<std::pair<int32, int32> > *submat_histogram_vector,
    // a vector of lists of pairs with rearranged pairs 
    std::vector<std::vector<std::pair<int32, int32> > > *split_lists
    )  {

  KALDI_ASSERT(max_submat_list_size > 0); 
  // sort the submat_histogram_vector based on second element of pair
  // in descending order then first element of pair in descending order
  std::sort(submat_histogram_vector->begin(),
            submat_histogram_vector->end(), SecondElementComparator);

  int32 prev_max_remaining_submat_list_size = max_submat_list_size;
  std::vector<std::pair<int32, int32> >::iterator iter;
  for (iter = submat_histogram_vector->begin();
       iter != submat_histogram_vector->end();
       iter++)  {
    std::pair<int32, int32> submat_index_and_count = *iter;
    std::vector<std::vector<std::pair<int32, int32> >::iterator>
        output_iterator_list;
    int32 max_remaining_submat_list_size = 0;
    FindSubmatIndexInSubmatLists(submat_index_and_count.first,
                                 sorted_submat_lists,
                                 &output_iterator_list,
                                 &max_remaining_submat_list_size);
    if (max_remaining_submat_list_size
        < prev_max_remaining_submat_list_size)  {
      // since we will have a smaller max_remaining_submat_list_size by
      // splitting this submat_index into a seperate list,
      // we will split it;
      std::vector<std::pair<int32, int32> > list_of_pairs;
      ExtractGivenPairsFromSubmatLists(output_iterator_list,
                                  sorted_submat_lists,
                                  &list_of_pairs);
      split_lists->push_back(list_of_pairs);
      prev_max_remaining_submat_list_size = max_remaining_submat_list_size;
    }
  }
  
  // rearrange the remaining pairs into lists where
  // pairs with multiple first elements are allowed 
  // Note : we don't yet know if there is any advantage of having multiple
  // calls to the same submat in kAddRowsMulti. If this is actually helpful
  // then use the sorted_histogram_vector to first copy submat_indexes which
  // did not make it to kAddRows calls
  for (int32 i = 0; i < prev_max_remaining_submat_list_size; i++) {
    std::vector<std::pair<int32, int32> > list_of_pairs;
    ExtractLastPairFromSubmatLists(sorted_submat_lists,
                                  &list_of_pairs);
    split_lists->push_back(list_of_pairs);
  }
}

// Function rearranges the submat_lists (see nnet-compute-utils.h for
// description of submat_lists), into lists that can be used as inputs
// for kAddRows and kAddRowsMulti calls.
// kAddRows requires a list of pairs where all the first elements correspond to
// the same submat_index. 
// kAddRowsMulti uses a list of pairs where the first elements can correspond to
// multiple submat_index locations.
// The maximum size of submat_lists is the minimum number of kAddRows* calls 
// necessary. In the current implementation we replace kAddRowsMulti calls with
// kAddRows calls wherever possible, while not increasing the number of calls.
//
// Algorithm : 
// The function computes a histogram of submat_indexes and spans through the
// submat_indexes in descending order of frequency. For each submat_index a
// decision is made to copy it using a kAddRows call or not.
// A kAddRow call is made for a submat_index if splitting it into a seperate
// list reduces the max_submat_list_size by one, i.e., reduces the number of
// remaining kAddRowsMulti calls.
// submat_indexes which cannot be assigned to kAddRow calls are rearranged into
// lists for kAddRowsMulti calls.
//
// Note : To decide splits we could have solved a combinatorial 
// optimization problem where we find the best set of
// kAddRows + kAddRowsMulti calls;
// but given that both these calls have similar costs,
// and that the average number of elements in a submat_list is around 4,
// it does not make sense to
// choose a kAddRows call unless it is able to immediately reduce a
// kAddRowsMulti call. So we simplify the process and stay away from any
// complex search algorithms. We might implement a solution where a more
// elaborate search is done,if the length of the lists increases
// for newer NN architectures, as even minor savings in speed due to increased
// number of kAddRows calls can accumulate compensating for the additional calls

void SplitLocations(
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    std::vector<std::vector<std::pair<int32, int32> > > *split_lists) {

  // a histogram of the submat_indexes in the submat_lists
  // each occurence in a given submat_list is considered unique so we maintain
  // a vector to count each occurence seperately
  unordered_map<int32, std::vector<int32> > submat_histogram;

  int32 max_submat_list_size = 0;

  // initializing a vector of list of pairs which is mutable
  // and where the sorted submat_lists are sorted, for faster search
  std::vector<std::vector< std::pair<int32, int32> > >
      sorted_submat_lists;
  sorted_submat_lists.reserve(submat_lists.size());
  ComputeSubmatIndexHistogram(submat_lists, &sorted_submat_lists,
                              &submat_histogram, &max_submat_list_size);
  // the vector has same information as the submat_histogram, but it is
  // suitable for sorting according to frequency. The first elements of pairs
  // can be repeated, these correspond to different occurrences in the same list 
  std::vector<std::pair<int32, int32> > submat_histogram_vector;
  // copy the key, occurence_counts from submat_histogram to a vector
  unordered_map<int32, std::vector<int32> >::iterator hist_iter;
  for (hist_iter = submat_histogram.begin();
       hist_iter != submat_histogram.end();
       ++hist_iter) {
    for (int32 i = 0; i < (hist_iter->second).size(); i++)  {
      submat_histogram_vector.push_back(
          std::make_pair(hist_iter->first, (hist_iter->second)[i]));
    }
  }
  SplitLocationsUsingSubmatHistogram(max_submat_list_size, &sorted_submat_lists,
                               &submat_histogram_vector, split_lists);
}

bool ConvertToIndexes(
    const std::vector<std::pair<int32, int32> > &location_vector,
    int32 *first_value,
    std::vector<int32> *second_values)  {
  *first_value = -1;
  second_values->reserve(location_vector.size());
  std::vector<std::pair<int32, int32> >::const_iterator iter;
  for (iter = location_vector.begin(); iter < location_vector.end(); iter++)  {
    if (iter->first != -1) {
      if (*first_value == -1)
        *first_value = iter->first;
      if (iter->first != *first_value)
        return false;
      second_values->push_back(iter->second);
    } else  {
      second_values->push_back(-1);
    }
  }
  return true;
}

}  // namespace nnet3
}  // namespace kaldi
