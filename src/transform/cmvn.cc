// transform/cmvn.cc

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

#include "transform/cmvn.h"

namespace kaldi {

void InitCmvnStats(int32 dim, Matrix<double> *stats) {
  KALDI_ASSERT(dim > 0);
  stats->Resize(2, dim+1);
}

void AccCmvnStats(const VectorBase<BaseFloat> &feats, BaseFloat weight, MatrixBase<double> *stats) {
  int32 dim = feats.Dim();
  KALDI_ASSERT(stats != NULL);
  KALDI_ASSERT(stats->NumRows() == 2 && stats->NumCols() == dim + 1);
  // Remove these __restrict__ modifiers if they cause compilation problems.
  // It's just an optimization.
   double *__restrict__ mean_ptr = stats->RowData(0),
       *__restrict__ var_ptr = stats->RowData(1),
       *__restrict__ count_ptr = mean_ptr + dim;
   const BaseFloat * __restrict__ feats_ptr = feats.Data();
  *count_ptr += weight;
  // Careful-- if we change the format of the matrix, the "mean_ptr < count_ptr"
  // statement below might become wrong.
  for (; mean_ptr < count_ptr; mean_ptr++, var_ptr++, feats_ptr++) {
    *mean_ptr += *feats_ptr * weight;
    *var_ptr +=  *feats_ptr * *feats_ptr * weight;
  }
}

void AccCmvnStats(const MatrixBase<BaseFloat> &feats,
                  const VectorBase<BaseFloat> *weights,
                  MatrixBase<double> *stats) {
  int32 num_frames = feats.NumRows();
  if (weights != NULL) {
    KALDI_ASSERT(weights->Dim() == num_frames);
  }
  for (int32 i = 0; i < num_frames; i++) {
    SubVector<BaseFloat> this_frame = feats.Row(i);
    BaseFloat weight = (weights == NULL ? 1.0 : (*weights)(i));
    if (weight != 0.0)
      AccCmvnStats(this_frame, weight, stats);
  }
}

void ApplyCmvn(const MatrixBase<double> &stats,
               bool var_norm,
               MatrixBase<BaseFloat> *feats) {
  KALDI_ASSERT(feats != NULL);
  int32 dim = stats.NumCols() - 1;
  if (stats.NumRows() > 2 || stats.NumRows() < 1 || feats->NumCols() != dim) {
    KALDI_ERR << "Dim mismatch: cmvn "
              << stats.NumRows() << 'x' << stats.NumCols()
              << ", feats " << feats->NumRows() << 'x' << feats->NumCols();
  }
  if (stats.NumRows() == 1 && var_norm)
    KALDI_ERR << "You requested variance normalization but no variance stats "
              << "are supplied.";
  
  double count = stats(0, dim);
  // Do not change the threshold of 1.0 here: in the balanced-cmvn code, when
  // computing an offset and representing it as stats, we use a count of one.
  if (count < 1.0)
    KALDI_ERR << "Insufficient stats for cepstral mean and variance normalization: "
              << "count = " << count;
  
  Matrix<BaseFloat> norm(2, dim);  // norm(0, d) = mean offset
  // norm(1, d) = scale, e.g. x(d) <-- x(d)*norm(1, d) + norm(0, d).
  for (int32 d = 0; d < dim; d++) {
    double mean, offset, scale;
    mean = stats(0, d)/count;
    if (!var_norm) {
      scale = 1.0;
      offset = -mean;
    } else {
      double var = (stats(1, d)/count) - mean*mean,
          floor = 1.0e-20;
      if (var < floor) {
        KALDI_WARN << "Flooring cepstral variance from " << var << " to "
                   << floor;
        var = floor;
      }
      scale = 1.0 / sqrt(var);
      if (scale != scale || 1/scale == 0.0)
        KALDI_ERR << "NaN or infinity in cepstral mean/variance computation";
      offset = -(mean*scale);
    }
    norm(0, d) = offset;
    norm(1, d) = scale;
  }
  // Apply the normalization.
  if (var_norm)
    feats->MulColsVec(norm.Row(1));
  feats->AddVecToRows(1.0, norm.Row(0));
}

void ApplyCmvnReverse(const MatrixBase<double> &stats,
                      bool var_norm,
                      MatrixBase<BaseFloat> *feats) {
  KALDI_ASSERT(feats != NULL);
  int32 dim = stats.NumCols() - 1;
  if (stats.NumRows() > 2 || stats.NumRows() < 1 || feats->NumCols() != dim) {
    KALDI_ERR << "Dim mismatch: cmvn "
              << stats.NumRows() << 'x' << stats.NumCols()
              << ", feats " << feats->NumRows() << 'x' << feats->NumCols();
  }
  if (stats.NumRows() == 1 && var_norm)
    KALDI_ERR << "You requested variance normalization but no variance stats "
              << "are supplied.";
  
  double count = stats(0, dim);
  // Do not change the threshold of 1.0 here: in the balanced-cmvn code, when
  // computing an offset and representing it as stats, we use a count of one.
  if (count < 1.0)
    KALDI_ERR << "Insufficient stats for cepstral mean and variance normalization: "
              << "count = " << count;
  
  Matrix<BaseFloat> norm(2, dim);  // norm(0, d) = mean offset
  // norm(1, d) = scale, e.g. x(d) <-- x(d)*norm(1, d) + norm(0, d).
  for (int32 d = 0; d < dim; d++) {
    double mean, offset, scale;
    mean = stats(0, d) / count;
    if (!var_norm) {
      scale = 1.0;
      offset = mean;
    } else {
      double var = (stats(1, d)/count) - mean*mean,
          floor = 1.0e-20;
      if (var < floor) {
        KALDI_WARN << "Flooring cepstral variance from " << var << " to "
                   << floor;
        var = floor;
      }
      // we aim to transform zero-mean, unit-variance input into data
      // with the given mean and variance.
      scale = sqrt(var);
      offset = mean;
    }
    norm(0, d) = offset;
    norm(1, d) = scale;
  }
  if (var_norm)
    feats->MulColsVec(norm.Row(1));
  feats->AddVecToRows(1.0, norm.Row(0));
}


void FakeStatsForSomeDims(const std::vector<int32> &dims,
                          MatrixBase<double> *stats) {
  KALDI_ASSERT(stats->NumRows() == 2 && stats->NumCols() > 1);
  int32 dim = stats->NumCols() - 1;
  double count = (*stats)(0, dim);
  for (size_t i = 0; i < dims.size(); i++) {
    int32 d = dims[i];
    KALDI_ASSERT(d >= 0 && d < dim);
    (*stats)(0, d) = 0.0;
    (*stats)(1, d) = count;
  }
}



}  // namespace kaldi
