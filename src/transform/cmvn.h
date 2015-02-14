// transform/cmvn.h

// Copyright 2009-2013 Microsoft Corporation
//                     Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_TRANSFORM_CMVN_H_
#define KALDI_TRANSFORM_CMVN_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"

namespace kaldi {

/// This function initializes the matrix to dimension 2 by (dim+1);
/// 1st "dim" elements of 1st row are mean stats, 1st "dim" elements
/// of 2nd row are var stats, last element of 1st row is count,
/// last element of 2nd row is zero.
void InitCmvnStats(int32 dim, Matrix<double> *stats);

/// Accumulation from a single frame (weighted).
void AccCmvnStats(const VectorBase<BaseFloat> &feat,
                  BaseFloat weight,
                  MatrixBase<double> *stats);

/// Accumulation from a feature file (possibly weighted-- useful in excluding silence).
void AccCmvnStats(const MatrixBase<BaseFloat> &feats,
                  const VectorBase<BaseFloat> *weights,  // or NULL
                  MatrixBase<double> *stats);

/// Apply cepstral mean and variance normalization to a matrix of features.
/// If norm_vars == true, expects stats to be of dimension 2 by (dim+1), but
/// if norm_vars == false, will accept stats of dimension 1 by (dim+1); these
/// are produced by the balanced-cmvn code when it computes an offset and
/// represents it as "fake stats".
void ApplyCmvn(const MatrixBase<double> &stats,
               bool norm_vars,
               MatrixBase<BaseFloat> *feats);

/// This is as ApplyCmvn, but does so in the reverse sense, i.e. applies a transform
/// that would take zero-mean, unit-variance input and turn it into output with the
/// stats of "stats".  This can be useful if you trained without CMVN but later want
/// to correct a mismatch, so you would first apply CMVN and then do the "reverse"
/// CMVN with the summed stats of your training data.
void ApplyCmvnReverse(const MatrixBase<double> &stats,
                      bool norm_vars,
                      MatrixBase<BaseFloat> *feats);


/// Modify the stats so that for some dimensions (specified in "dims"), we
/// replace them with "fake" stats that have zero mean and unit variance; this
/// is done to disable CMVN for those dimensions.
void FakeStatsForSomeDims(const std::vector<int32> &dims,
                          MatrixBase<double> *stats);
                          


}  // namespace kaldi

#endif  // KALDI_TRANSFORM_CMVN_H_
