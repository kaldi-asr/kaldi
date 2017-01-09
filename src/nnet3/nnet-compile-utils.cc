// nnet3/nnet-compile-utils.cc

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

// This comparator is used with std::find_if function to search for pairs
// whose first element is equal to the given pair
struct FirstElementIsEqualComparator :
      public std::unary_function<std::pair<int32, int32>, bool>
{
  explicit FirstElementIsEqualComparator(const int32 element):
      element_(element) {}
  bool operator() (std::pair<int32, int32> const &arg)
  { return (arg.first == element_); }
  int32 element_;
};

// This comparator is used with std::find_if function to search for pairs
// whose .first and .second elements are equal to the given pair
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

// this comparator will be used to sort pairs initially by second element in
// descending order and then by first element in descending order.
// note, std::sort accepts an actual function as an alternative to a
// function object.
bool  SecondElementComparator(const std::pair<int32, int32>& first_pair,
                              const std::pair<int32, int32>& second_pair) {
  if (first_pair.second == second_pair.second)
    return first_pair.first > second_pair.first;
  return first_pair.second > second_pair.second;
}

// Function to sort the lists in a vector of lists of pairs, by the first
// element of the pair
void SortSubmatLists(
    // vector of list of location pairs
    const std::vector<std::vector<std::pair<int32, int32> > > submat_lists,
    // a copy of the input submat_lists where the lists are sorted
    // (this will be used in the caller function for sort and find functions)
    std::vector<std::vector<std::pair<int32, int32> > > * sorted_submat_lists,
    // maximum size of the submat_lists
    int32* max_submat_list_size
    )
{
  *max_submat_list_size = 0;
  sorted_submat_lists->reserve(submat_lists.size());
  KALDI_ASSERT(submat_lists.size() > 0);
  for (int32 i = 0; i < submat_lists.size(); i++) {
    if (submat_lists[i].size() > *max_submat_list_size)
      *max_submat_list_size = submat_lists[i].size();
    sorted_submat_lists->push_back(submat_lists[i]);
    std::sort((*sorted_submat_lists)[i].begin(),
              (*sorted_submat_lists)[i].end(),
              FirstElementComparator());
  }
}

// Function to compute a histogram of the submat_index,
// which is the first_element in the location pair, given vector of list of
// location pairs
void ComputeSubmatIndexHistogram(
    // vector of list of pairs of location pairs where the lists are sorted
    // by submat_indexes (.first element)
    const std::vector<std::vector<std::pair<int32, int32> > >
    sorted_submat_lists,
    // a histogram of submat_indexes where
    // the keys are submat_indexes and values are a vector of frequencies
    // of first occurrence, second occurrence, etc. of a submat_index
    // in a submat_list
    unordered_map<int32, std::vector<int32> >* submat_histogram
    ) {
  KALDI_ASSERT(sorted_submat_lists.size() > 0);
  // computing the submat_histogram
  // counting the occurrences of each element in the current submat_list;
  // each new occurrence of the same element, in this list, is counted
  // as a seperate symbol for frequency counts
  for (int32 i = 0; i < sorted_submat_lists.size(); i++) {
    int j = 0;
    unordered_map<int32, std::vector<int32> >::iterator histogram_iterator
        = submat_histogram->end();
    int32 repetition_count = 0;
    while (j < sorted_submat_lists[i].size()) {
      if ((histogram_iterator == submat_histogram->end()) ||
          (histogram_iterator->first != sorted_submat_lists[i][j].first)) {
        histogram_iterator =
            submat_histogram->find(sorted_submat_lists[i][j].first);
        repetition_count = 0;
        // if a histogram entry was not found for this submat_index, add one
        if (histogram_iterator == submat_histogram->end()) {
          (*submat_histogram)[sorted_submat_lists[i][j].first];
          histogram_iterator = submat_histogram->find(
              sorted_submat_lists[i][j].first);
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
    // submat_index to search in the submat_lists
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
        (*sorted_submat_lists)[i];
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

// Function to extract the identified pairs (identified with an iterator)
// from a vector of list of pairs, "to extract" means to copy into
// a list and erase the original pair from the submat_lists
void ExtractGivenPairsFromSubmatLists(
    std::vector<std::vector<std::pair<int32, int32> >::iterator>
    &input_iterator_list,
    std::vector<std::vector<std::pair<int32, int32> > > *sorted_submat_lists,
    std::vector<std::pair<int32, int32> > *list_of_pairs) {
  list_of_pairs->reserve(sorted_submat_lists->size());
  for (int32 i = 0; i < input_iterator_list.size(); i++) {
    if (input_iterator_list[i] != (*sorted_submat_lists)[i].end()) {
      // there was an element with the submat_index in the current list
      list_of_pairs->push_back(*input_iterator_list[i]);
      (*sorted_submat_lists)[i].erase(input_iterator_list[i]);
    } else  {
      // insert a dummy element. Callers of this function expect the dummy
      // element to be (-1, -1)
      list_of_pairs->push_back(std::make_pair(-1, -1));
    }
  }
}

// Function to extract the last pairs from a vector of list of pairs
// a dummy is added when the list is empty
static void ExtractLastPairFromSubmatLists(
    std::vector<std::vector<std::pair<int32, int32> > > *sorted_submat_lists,
    std::vector<std::pair<int32, int32> > *list_of_pairs) {
  list_of_pairs->reserve(sorted_submat_lists->size());
  for (int32 i = 0; i < sorted_submat_lists->size(); i++) {
    if ((*sorted_submat_lists)[i].size() == 0)  {
      // the value of the dummy has to be (-1, -1) as down stream code has
      // expects -1 values for dummies
      list_of_pairs->push_back(std::make_pair(-1, -1));
      continue;
    }
    list_of_pairs->push_back((*sorted_submat_lists)[i].back());
    (*sorted_submat_lists)[i].pop_back();
  }
}

// Function which does the actual splitting of submat_lists. But it operates on
// sorted submat_lists and uses submat_histogram_vector.
// See SplitLocations, below for the algorithm
static void SplitLocationsUsingSubmatHistogram(
    // maximum size of the lists in the sorted_submat_lists
    int32 max_submat_list_size,
    // a vector of list of pairs where each list is expected to be sorted
    // this is a pointer as the lists will be modified
    std::vector<std::vector<std::pair<int32, int32> > > *sorted_submat_lists,
    // a vector of pairs to represent a histogram
    // this is a pointer as the vector will be sorted
    std::vector<std::pair<int32, int32> > *submat_histogram_vector,
    // a vector of lists of pairs with rearranged pairs
    std::vector<std::vector<std::pair<int32, int32> > > *split_lists)  {

  // sort the submat_histogram_vector based on second element of pair
  // in descending order then first element of pair in descending order
  std::sort(submat_histogram_vector->begin(),
            submat_histogram_vector->end(), SecondElementComparator);

  int32 prev_max_remaining_submat_list_size = max_submat_list_size;
  std::vector<std::pair<int32, int32> >::iterator iter;
  for (iter = submat_histogram_vector->begin();
       iter != submat_histogram_vector->end();
       ++iter)  {
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
// ------------------------
// The maximum size of a list in submat_lists is the minimum number of
// kAddRowsMulti calls necessary.
// In the current implementation we replace kAddRowsMulti calls with
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
  // a vector to count each occurrence separately.
  // The i'th element in the vector corresponds to the count of
  // the (i+1)'th occurrence of a submat_index in a submat_list
  unordered_map<int32, std::vector<int32> > submat_histogram;

  int32 max_submat_list_size = 0;

  // initializing a vector of list of pairs to store the sorted submat_lists
  std::vector<std::vector< std::pair<int32, int32> > >
      sorted_submat_lists;
  SortSubmatLists(submat_lists, &sorted_submat_lists, &max_submat_list_size);
  ComputeSubmatIndexHistogram(sorted_submat_lists, &submat_histogram);
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

/* If it is the case for some i >= 0 that all the .first elements of
   "location_vector" are either i or -1, then output i to first_value and the
   .second elements into "second_values", and return true.  Otherwise return
   false and the outputs are don't-cares. */
bool ConvertToIndexes(
    const std::vector<std::pair<int32, int32> > &location_vector,
    int32 *first_value,
    std::vector<int32> *second_values)  {
  *first_value = -1;
  second_values->clear();
  second_values->reserve(location_vector.size());
  std::vector<std::pair<int32, int32> >::const_iterator iter;
  for (iter = location_vector.begin(); iter < location_vector.end(); ++iter)  {
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


// see declaration in header for documentation
void EnsureContiguousProperty(
    const std::vector<int32> &indexes,
    std::vector<std::vector<int32> > *indexes_out) {
  indexes_out->clear();
  indexes_out->reserve(3);
  if (indexes.empty()) return;
  int32 max_value = *std::max_element(indexes.begin(), indexes.end());
  if (max_value == -1) return;
  std::vector<int32> num_segments_seen(max_value + 1, 0);
  int32 dim = indexes.size(), num_output_vectors = 0;
  for (int32 i = 0; i < dim;) {
    // note, we increment i within the loop.
    if (indexes[i] == -1) {
      i++;
      continue;
    }
    int32 value = indexes[i], start_index = i;
    for (; i < dim && indexes[i] == value; i++);
    int32 end_index = i;  // one past the end.
    // the input 'indexes' contains a sequence of possibly-repeated instances of
    // the value 'value', starting at index 'start_index', with 'end_index' as
    // one past the end.
    int32 this_num_segments_seen = num_segments_seen[value]++;
    if (this_num_segments_seen >= num_output_vectors) {  // we have nowhere to
                                                         // put it.
      indexes_out->resize(++num_output_vectors);
      indexes_out->back().resize(dim, -1);  // fill newly added vector with -1's.
    }
    std::vector<int32> &this_out_vec((*indexes_out)[this_num_segments_seen]);
    std::vector<int32>::iterator iter = this_out_vec.begin() + start_index,
        end = this_out_vec.begin() + end_index;
    // Fill the appropriate range of the output vector with 'value'
    for (; iter != end; ++iter) *iter = value;
  }
}



/**
   This function splits a vector of pairs into a list of vectors of pairs.
   [note: by 'vector' we mean something that has a meaningful index that we care
   about; by 'list' we mean a collection of elements to be iterated over, without
   (in this case) meaningful indexes or even order.

   @param [in] list   A vector of pairs; these pairs should be either (-1,-1)
                      or (a,b) for a >= 0, b >= 0.  At least one element of 'list'
                      must be different from (-1,-1).
   @param [out] split_lists   A list, in arbitrary order, of vectors of pairs.
                     It has the following relationship with 'list':
                     - Size: for each j, split_lists[j].size() == list.size().
                     - Contents must match input: For each i:
                       -  If list[i] == (-1, -1), then
                          split_lists[j][i] == (-1, -1) for all j.
                       -  If list[i] != (-1, -1), then
                         split_lists[j][i] == (-1, -1) for *all but one* j, and
                         for the remaining j, split_lists[j][i] == list[i].
                     - Uniqueness: for no j should split_lists[j] contain
                       any duplicate elements (except the pair (-1,-1), which  is
                       allowed to exist in duplicate form).
                     To satisfy the above conditions, this function will create
                     as many lists in split_lists (i.e. as many j values) as the
                     number of times that the most frequent pair in 'list'
                     repeats other than the pair (-1,-1), e.g. if the pair
                     (10,11) appears 4 times in 'list' and that is the most,
                     split_lists->size() == 4.
*/
void SplitPairList(std::vector<std::pair<int32, int32> >& list,
                   std::vector<std::vector<std::pair<int32, int32> > >* split_lists) {
  split_lists->clear();
  typedef unordered_map<std::pair<int32, int32>,
                        int32, PairHasher<int32> > MapType;
  // this maps a pair not equal to -1,-1, to the number of times we've already seen it.
  MapType pair_to_count;
  int32 cur_num_lists = 0;

  for (int32 i = 0; i < list.size(); i++)  {
    if (list[i].first == -1)
      continue;
    MapType::iterator iter = pair_to_count.find(list[i]);
    int32 this_count;
    if (iter == pair_to_count.end())
      pair_to_count[list[i]] = this_count = 1;
    else
      this_count = (++iter->second);
    if (this_count > cur_num_lists) {
      KALDI_ASSERT(this_count == cur_num_lists + 1);
      split_lists->resize(this_count);
      split_lists->back().resize(list.size(),
                                 std::pair<int32, int32>(-1, -1));
      cur_num_lists++;
    }
    (*split_lists)[this_count-1][i] = list[i];
  }
  if (split_lists->size() == 0)
    KALDI_ERR << "Input list has just dummy pairs";
}

void SplitLocationsBackward(
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    std::vector<std::vector<std::pair<int32, int32> > > *split_lists) {
  std::vector<std::vector<std::pair<int32, int32> > > split_lists_intermediate;
  // Split the submat_lists
  SplitLocations(submat_lists, &split_lists_intermediate);
  for (size_t i = 0; i < split_lists_intermediate.size(); i++) {
    int32 first_value;
    std::vector<int32> second_values;
    if (ConvertToIndexes(split_lists_intermediate[i],
                         &first_value, &second_values)) {
      // the .first values in split_lists_intermediate[i] are all the same (or
      // equal to -1).
      if (first_value == -1) {
        // all the .first values were equal to -1.  this is like a NULL marker.
        continue;
      }
      std::vector<std::vector<int32> > second_values_split;
      EnsureContiguousProperty(second_values, &second_values_split);
      if (second_values_split.size() == 1) {
        // this branch is an optimization for speed.
        split_lists->push_back(split_lists_intermediate[i]);
      } else {
        for (size_t j = 0; j < second_values_split.size(); j++) {
          split_lists->resize(split_lists->size() + 1);
          const std::vector<int32> &input_list = second_values_split[j];
          std::vector<std::pair<int32, int32> > &output_list =
              split_lists->back();
          output_list.resize(input_list.size());
          int32 size = input_list.size();
          for (int32 k = 0; k < size; k++) {
            int32 row = input_list[k];
            if (row == -1) output_list[k].first = -1;
            else output_list[k].first = first_value;
            output_list[k].second = row;
          }
        }
      }
    } else {
      // the .first values are not the same
      // splitting the list of pairs to ensure unique pairs, unless it is
      // (-1,-1)
      std::vector<std::vector<std::pair<int32, int32> > > new_split_lists;
      SplitPairList(split_lists_intermediate[i],
                    &new_split_lists);
      for (int32 j = 0; j < new_split_lists.size(); j++)  {
        split_lists->push_back(new_split_lists[j]);
      }
    }
  }
}

// This function returns true if for each integer i != -1, all the indexes j at
// which indexes[j] == i are consecutive with no gaps (more formally: if j1 < j2
// < j3 and indexes[j1] == indexes[j3], then indexes[j1] == indexes[j2]).  If
// so, it also outputs to "reverse_indexes" the begin and end of these ranges,
// so that indexes[j] == i for all j such that (*reverse_indexes)[i].first <= j
// && j < (*reverse_indexes)[i].second.
bool HasContiguousProperty(
    const std::vector<int32> &indexes,
    std::vector<std::pair<int32, int32> > *reverse_indexes) {
  reverse_indexes->clear();
  int32 num_indexes = indexes.size();
  if (num_indexes == 0)
    return true;
  int32 num_input_indexes =
      *std::max_element(indexes.begin(), indexes.end()) + 1;
  KALDI_ASSERT(num_input_indexes >= 0);
  if (num_input_indexes == 0) {
    // we don't really expect this input, filled with -1's.
    KALDI_WARN << "HasContiguousProperty called on vector of -1's.";
    return true;
  }
  reverse_indexes->resize(num_input_indexes,
                          std::pair<int32,int32>(-1, -1));
  // set each pair's "first" to the min index of all elements
  // of "indexes" with that value, and the "second" to the
  // max plus one.
  for (int32 i = 0; i < num_indexes; i++) {
    int32 j = indexes[i];
    if (j == -1) continue;
    KALDI_ASSERT(j >= 0);
    std::pair<int32, int32> &pair = (*reverse_indexes)[j];
    if (pair.first == -1) {
      pair.first = i;
      pair.second = i + 1;
    } else {
      pair.first = std::min(pair.first, i);
      pair.second = std::max(pair.second, i + 1);
    }
  }
  // check that the contiguous property holds.
  for (int32 i = 0; i < num_input_indexes; i++) {
    std::pair<int32, int32> pair = (*reverse_indexes)[i];
    if (pair.first != -1) {
      for (int32 j = pair.first; j < pair.second; j++)
        if (indexes[j] != i)
          return false;
    }
  }
  return true;
}

}  // namespace nnet3
}  // namespace kaldi
