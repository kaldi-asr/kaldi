// nnet3/nnet-compile-utils.h

// Copyright 2015    Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_COMPILE_UTILS_H_
#define KALDI_NNET3_NNET_COMPILE_UTILS_H_

#include "nnet3/nnet-component-itf.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-computation-graph.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {
// this file contains some utility functions used in nnet-compile.h


/**
   The input to this function is a vector of lists of pairs, and this function
   splits it up into a list of vectors of pairs.  In order to make the lists all
   the same length it may have to insert "dummy" pairs with value (-1, -1).
   In addition, this function implement certain heuristics to break up the
   list into pairs in a particular desirable way, as we will describe below.

   submat_lists.dim() may be large e.g. 1024 (it usually represents a minibatch
   size), but the maximum size of the lists will usually be fairly small e.g. no
   more than 4 or so, as it represents the number of terms in a hand-coded
   summation expression.

   The use of this function is in interpreting a command to set each row of
   a matrix to a sum of terms.  Each pair represents an input term, interpreted
   as (index-of-matrix, row-index), which represents a vector that will form
   part of the sum.

   It would be possible to simply pad at the end with (-1, -1), but this
   function also makes an attempt to pad more carefully so that for the most
   part, each output vector of pairs has inputs from only one matrix, i.e.  the
   pair.first values are all the same.  This will allow us to use a potentially
   more efficient command in the compiled code.  It doesn't have to be 100%
   optimal.  Note: in the most common case, all the lists will have the same
   length and padding will not be necessary at all.

   See documentation here: \ref dnn3_compile_compiler_split_locations
 */

void SplitLocations(
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    std::vector<std::vector<std::pair<int32, int32> > > *split_lists);


/**
   This function has the same interface as SplitLocations(); however, it ensures
   certain additional properties of the output "split_lists", which are necessary
   because of the way it is used in backprop code.
   For each sub-list sublist = (*split_lists)[i], the properties it ensures are:
      Either:
        - all pairs in the list "sublist" are unique (except that the special pair
          (-1, -1) may be repeated),
      Or:
        - the .first values in the list "sublist" are all the same, and the
          .second have a special property [see the function HasContiguousProperty
          in nnet-compile.cc]- basically that if we list the .second elements,
          each unique number that appears there appears only in one contiguous range,
          e.g. the list [ 6 6 6 1 5 5 5 ] has this property, but [ 1 2 1 ] does not.
          (however, -1's are not subject to this limitation, so [ -1 4 -1 ] satisfies
          the property).
  This function ensures this property by first calling SplitLocations, and then
  doing further splitting as necessary to ensure the property.  However, if as a result
  it needs to split any given initially-split list into more than 2 sub-lists, it will
  print a warning (once per process).  If we have to split into too many lists it
  will generate inefficient computations, and we will need to extend the backprop
  code to support more general types of operation.
  If all elements of submat_lists are empty, the output split_lists will be
  the empty vector.
 */
void SplitLocationsBackward(
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    std::vector<std::vector<std::pair<int32, int32> > > *split_lists);


/** If it is the case for some i >= 0 that all the .first elements of
   "location_vector" are either i or -1, then output i to first_value and the
   .second elements into "second_values", and return true.  Otherwise return
   false and the outputs are don't-cares. */
bool ConvertToIndexes(
    const std::vector<std::pair<int32, int32> > &location_vector,
    int32 *first_value,
    std::vector<int32> *second_values);

/** This function returns true if for each integer i != -1, all the indexes j at
    which indexes[j] == i are consecutive with no gaps (more formally: if j1 <
    j2 < j3 and indexes[j1] != -1 and indexes[j1] == indexes[j3], then
    indexes[j1] == indexes[j2]).  For example, the vector [ 1 2 1 ] lacks the
    contiguous property because 1 appears in two places with a different number
    in the middle.  If the vector has the contiguous property, this function
    also outputs to "reverse_indexes" the begin and end of these ranges, so that
    indexes[j] == i for all j such that (*reverse_indexes)[i].first <= j && j <
    (*reverse_indexes)[i].second. */
bool HasContiguousProperty(const std::vector<int32> &indexes,
                           std::vector<std::pair<int32, int32> > *reverse_indexes);


/** This function takes a vector of indexes and splits it up into as separate
    vectors of the same size, as needed to ensure that the 'contiguous property' holds.
    This is done via padding with -1's.  An example will clarify this.  Suppose the
    input is:
      [ -1  1  1  1  2  2  1  1 ]
    which lacks the contiguous property because 1's appear in 2 different places, it
    would split it up as
      [ -1  1  1  1  2  2 -1 -1 ]
      [ -1 -1 -1 -1 -1 -1  1  1 ]
    If 'indexes' is empty or only contains -1's, 'indexes_out' will be empty.
 */
void EnsureContiguousProperty(
    const std::vector<int32> &indexes,
    std::vector<std::vector<int32> > *indexes_out);



} // namespace nnet3
} // namespace kaldi


#endif

