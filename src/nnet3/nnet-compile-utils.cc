// nnet3/nnet-compile-utils.cc

// Copyright      2015-2017  Johns Hopkins University (author: Daniel Povey)
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

/**
   Gets counts of submatrices (the 1st members of pairs) in submat_lists.
   Also outputs, to 'submats_with_large_counts', a list of submatrix indexes
   that have counts over half of submat_lists.size().  (These will be separated
   out into their own AddRows() commands).
 */
void GetSubmatCounts(
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    std::unordered_map<int32,int32> *submat_counts,
    std::vector<int32> *submats_with_large_counts) {
  auto iter = submat_lists.begin(), end = submat_lists.end();
  for (; iter != end; ++iter) {
    std::vector<std::pair<int32, int32> >::const_iterator
        iter2 = iter->begin(), end2 = iter->end();
    for (; iter2 != end2; ++iter2) {
      int32 submat_index = iter2->first;
      KALDI_ASSERT(submat_index >= 0);  // We don't expect -1's in submat_lists.
      std::unordered_map<int32,int32>::iterator
          iter = submat_counts->find(submat_index);
      if (iter == submat_counts->end())
        (*submat_counts)[submat_index] = 1;
      else
        iter->second++;
    }
  }
  auto counts_iter = submat_counts->begin(),
      counts_end = submat_counts->end();
  size_t cutoff = submat_lists.size() / 2;
  for (; counts_iter != counts_end; ++counts_iter)
    if (counts_iter->second > cutoff)
      submats_with_large_counts->push_back(counts_iter->first);
}

/**
   This function, used in SplitLocations(), is used to make separate
   'split lists' for certain high-count submatrix indexes, specified by
   the user in 'submats_to_separate'.  These split
   lists will be lists of pairs that are all either (-1, 1) or (submatrix_index, x)
   for a particular submatrix index (constant within the split list).
   These high-count lists will be written to 'split_lists'; they
   will eventually compile to AddRows() commands.  We write the remaining
   members of the lists in 'submat_lists' (the ones that did not make it
   into 'split_lists') to 'reduced_submat_lists'.
 */
void SeparateSubmatsWithLargeCounts(
    const std::vector<int32> &submats_to_separate,
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    std::vector<std::vector<std::pair<int32, int32> > > *reduced_submat_lists,
    std::vector<std::vector<std::pair<int32, int32> > > *split_lists) {
  KALDI_ASSERT(split_lists->empty() && !submats_to_separate.empty());
  size_t num_to_separate = submats_to_separate.size(),
      num_rows = submat_lists.size();
  std::unordered_map<int32, size_t> submat_to_index;
  reduced_submat_lists->clear();
  reduced_submat_lists->resize(num_rows);
  split_lists->resize(num_to_separate);
  for (size_t i = 0; i < num_to_separate; i++) {
    (*split_lists)[i].resize(num_rows, std::pair<int32, int32>(-1, -1));
    int32 submat = submats_to_separate[i];
    submat_to_index[submat] = i;
  }
  for (size_t row = 0; row < submat_lists.size(); row++) {
    std::vector<std::pair<int32, int32> >::const_iterator
        iter = submat_lists[row].begin(), end = submat_lists[row].end();
    std::vector<std::pair<int32, int32> >
        &reduced_list = (*reduced_submat_lists)[row];
    // 'reduced_lists' will contain the pairs that don't make it into
    // 'split_lists'.
    for (; iter != end; ++iter) {
      int32 submat_index = iter->first;
      std::unordered_map<int32, size_t>::const_iterator map_iter =
          submat_to_index.find(submat_index);
      if (map_iter == submat_to_index.end()) { // not a large-count submatrix.
        reduced_list.push_back(*iter);
        continue;
      }
      size_t index = map_iter->second;
      std::pair<int32,int32> &p = (*split_lists)[index][row];
      if (p.first >= 0) {
        // we'd only reach here if the same submat index repeated in the same
        // row, which is possible but rare.
        reduced_list.push_back(*iter);
        continue;
      }
      p.first = submat_index;
      int32 src_row_index = iter->second;
      p.second = src_row_index;
    }
  }
}

void SplitLocations(
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    std::vector<std::vector<std::pair<int32, int32> > > *split_lists) {
  size_t num_rows = submat_lists.size(),
      num_output_lists = 0;
  auto iter = submat_lists.begin(), end = submat_lists.end();
  for (; iter != end; ++iter)
    if (iter->size() > num_output_lists)
      num_output_lists = iter->size();
  split_lists->clear();
  if (num_output_lists == 0)  // Odd, but could happen, maybe
    return;
  else if (num_output_lists == 1) {
    split_lists->resize(1);
    std::vector<std::pair<int32, int32> > &list = (*split_lists)[0];
    list.resize(num_rows, std::pair<int32, int32>(-1, -1));
    for (size_t i = 0; i < num_rows; i++) {
      if (!submat_lists[i].empty())
        list[i] = submat_lists[i][0];
    }
    return;
  }

  // counts for each submatrix index, of how many times it occurs.
  std::unordered_map<int32,int32> submat_counts;
  std::vector<int32> submats_with_large_counts;
  GetSubmatCounts(submat_lists, &submat_counts, &submats_with_large_counts);
  if (!submats_with_large_counts.empty()) {
    // There are submatrices with counts over half the num-rows.  We assign these
    // their own output lists.

    std::vector<std::vector<std::pair<int32, int32> > > reduced_submat_lists;
    SeparateSubmatsWithLargeCounts(submats_with_large_counts,
                                   submat_lists,
                                   &reduced_submat_lists,
                                   split_lists);
    // 'reduced_split_lists' is the result of recursing with input 'reduced_submat_lists';
    // we'll append its result to 'split_lists'.
    std::vector<std::vector<std::pair<int32, int32> > > reduced_split_lists;
    SplitLocations(reduced_submat_lists, &reduced_split_lists);
    size_t cur_num_lists = split_lists->size(),
        num_extra_lists = reduced_split_lists.size(),
        new_num_lists = cur_num_lists + num_extra_lists;
    split_lists->resize(new_num_lists);
    for (size_t i = 0; i < num_extra_lists; i++)
      (*split_lists)[cur_num_lists + i].swap(reduced_split_lists[i]);
    return;
    // and we're done.
  } else {
    // All the counts of submatrix indexes seem to be small so we are resigned to
    // only using AddRowsMulti commands.
    split_lists->resize(num_output_lists);
    for (size_t i = 0; i < num_output_lists; i++)
      (*split_lists)[i].resize(num_rows, std::pair<int32, int32>(-1, -1));
    for (size_t row = 0; row < num_rows; row++) {
      const std::vector<std::pair<int32, int32> > &this_list =
          submat_lists[row];
      size_t this_list_size = submat_lists[row].size();
      for (size_t i = 0; i < this_list_size; i++) {
        (*split_lists)[i][row] = this_list[i];
      }
    }
  }
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


// see comment in header.
void GetNxList(const std::vector<Index> &indexes,
               std::vector<std::pair<int32, int32> > *pairs) {
  // set of (n,x) pairs
  std::unordered_set<std::pair<int32, int32>, PairHasher<int32> > n_x_set;

  for (std::vector<Index>::const_iterator iter = indexes.begin();
       iter != indexes.end(); ++iter)
    n_x_set.insert(std::pair<int32, int32>(iter->n, iter->x));
  pairs->clear();
  pairs->reserve(n_x_set.size());
  for (std::unordered_set<std::pair<int32, int32>, PairHasher<int32> >::iterator
           iter = n_x_set.begin(); iter != n_x_set.end(); ++iter)
    pairs->push_back(*iter);
  std::sort(pairs->begin(), pairs->end());
}


// see comment in header.
void GetTList(const std::vector<Index> &indexes,
              std::vector<int32> *t_values) {
  // set of t values
  std::unordered_set<int32> t_set;

  for (std::vector<Index>::const_iterator iter = indexes.begin();
       iter != indexes.end(); ++iter)
    if (iter->t != kNoTime)
      t_set.insert(iter->t);
  t_values->clear();
  t_values->reserve(t_set.size());
  for (std::unordered_set<int32>::iterator iter = t_set.begin();
       iter != t_set.end(); ++iter)
    t_values->push_back(*iter);
  std::sort(t_values->begin(), t_values->end());
}



}  // namespace nnet3
}  // namespace kaldi
