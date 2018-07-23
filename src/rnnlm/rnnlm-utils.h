// rnnlm/rnnlm-utils.h

// Copyright 2017  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_RNNLM_RNNLM_UTILS_H_
#define KALDI_RNNLM_RNNLM_UTILS_H_

#include "base/kaldi-common.h"
#include "matrix/sparse-matrix.h"

// This file is for miscellaneous function declarations needed for the RNNLM
// code.

namespace kaldi {
namespace rnnlm {


/**
   Reads a text file (e.g. exp/rnnlm/word_feats.txt) which maps words to sparse
   combinations of features.  The text file contains lines of the format:
     <word-index>  <feature-index1> <feature-value1> <feature-index2> <feature-value2>  ...
   with the feature-indexes in sorted order: for example,
     2056  11 3.0 25 1.0 1069 1.0
   The word-indexes are expected to be in order 0, 1, 2, ...; so they don't really
   add any information; they are included for human readability.

   This function will throw an exception if the input is not as expected.

     @param [in] is   The stream we are reading.
     @param [in] feature_dim  The feature dimension, which equals the highest-numbered
                               possible feature plus one.  We don't attempt to work this
                               out from the input, in case for some reason this vocabulary
                               does not use the highest-numbered feature.
     @param [out] word_feature_matrix   A sparse matrix of dimension num-words by
                               feature-dim, containing the information in the file
                               we read.
 */
void ReadSparseWordFeatures(std::istream &is,
                            int32 feature_dim,
                            SparseMatrix<BaseFloat> *word_feature_matrix);



} // namespace rnnlm
} // namespace kaldi

#endif //KALDI_RNNLM_RNNLM_H_
