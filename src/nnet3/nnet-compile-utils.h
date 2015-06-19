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

   submat_lists.dim() may be large e.g. 1024 (it usually represents a minibatch
   size), but the maximum size of the lists will usually be fairly small e.g. no
   more than 4 or so, as it represents the number of terms in a hand-coded
   summation expression.
   
   The use of this function is in interpreting a command to set each row of
   a matrix to a sum of terms.  Each pair represents an input term, interpreted
   as (index-of-matrix, row-index), which represents a vector that will form
   part of the sum.

   It would be possible to simply pad at the end with (-1, -1), but this function
   should also make an attempt to pad more carefully so that for the most
   part, each output vector of pairs has inputs from only one matrix, i.e.
   the .first values are all the same.  This will allow us to use a potentially
   more efficient command in the compiled code.  It doesn't have to be 100% optimal.
   Note: in the most common case, all the lists will have the same length
   and padding will not be necessary at all.
 */

void SplitLocations(
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    std::vector<std::vector<std::pair<int32, int32> > > *split_lists);


/* If it is the case for some i >= 0 that all the .first elements of
   "location_vector" are either i or -1, then output i to first_value and the
   .second elements into "second_values", and return true.  Otherwise return
   false and the outputs are don't-cares. */
bool ConvertToIndexes(
    const std::vector<std::pair<int32, int32> > &location_vector,
    int32 *first_value,    
    std::vector<int32> *second_values);




} // namespace nnet3
} // namespace kaldi


#endif

