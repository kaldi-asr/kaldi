// transform/minmax.cc

// Copyright 2013 IDIAP Research Institute (author: Blaise Potard)

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

#include "idlakfeat/minmax.h"

namespace kaldi {

void InitMinmaxStats(int32 dim, Matrix<double> *stats) {
  KALDI_ASSERT(dim > 0);
  stats->Resize(2, dim+1);
  double *__restrict__ min_ptr = stats->RowData(0);
  double *__restrict__ max_ptr = stats->RowData(1);
  int i;
  for (i = 0; i < dim; i++) {
      min_ptr[i] = HUGE_VAL;
      max_ptr[i] = -HUGE_VAL;
  }
  min_ptr[dim] = 0;
  max_ptr[dim] = 0;
}

void AccMinmaxStats(const VectorBase<BaseFloat> &feats, BaseFloat weight, MatrixBase<double> *stats) {
  int32 dim = feats.Dim();
  KALDI_ASSERT(stats != NULL);
  KALDI_ASSERT(stats->NumRows() == 2 && stats->NumCols() == dim + 1);
  // Remove these __restrict__ modifiers if they cause compilation problems.
  // It's just an optimization.
   double *__restrict__ min_ptr = stats->RowData(0),
       *__restrict__ max_ptr = stats->RowData(1),
       *__restrict__ count_ptr = min_ptr + dim;
   const BaseFloat * __restrict__ feats_ptr = feats.Data();
  *count_ptr += weight;
  // Careful-- if we change the format of the matrix, the "mean_ptr < count_ptr"
  // statement below might become wrong.
  for (; min_ptr < count_ptr; min_ptr++, max_ptr++, feats_ptr++) {
    double a = *feats_ptr * weight;
    if (a < *min_ptr) *min_ptr = a;
    if (a > *max_ptr) *max_ptr = a;
  }
}

void AccMinmaxStats(const MatrixBase<BaseFloat> &feats,
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
      AccMinmaxStats(this_frame, weight, stats);
  }
}

void ApplyMinmax(const MatrixBase<double> &stats,
		 bool var_norm,
		 MatrixBase<BaseFloat> *feats) {
  KALDI_ASSERT(feats != NULL);
  int32 dim = stats.NumCols() - 1;
  if (stats.NumRows() > 2 || stats.NumRows() < 1 || feats->NumCols() != dim) {
    KALDI_ERR << "Dim mismatch in ApplyCmvn: cmvn "
              << stats.NumRows() << 'x' << stats.NumCols()
              << ", feats " << feats->NumRows() << 'x' << feats->NumCols();
  }
  if (stats.NumRows() == 1)
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
  // TODO: make these values configurable as parameters
  double ca = 0.01;
  double da = 0.99;
  for (int32 d = 0; d < dim; d++) {
    double min, max, offset, scale;
    min = stats(0, d);
    max = stats(1, d);
    //if (!var_norm) {
    //scale = 1.0;
    offset = (max * ca - min * da) / (max - min);
    scale = (da - ca) / (max - min);
    norm(0, d) = offset;
    norm(1, d) = scale;
  }
  int32 num_frames = feats->NumRows();

  // Apply the normalization.
  for (int32 i = 0; i < num_frames; i++) {
    for (int32 d = 0; d < dim; d++) {
      BaseFloat &f = (*feats)(i, d);
      f = norm(0, d) + f*norm(1, d);
    }
  }
}


}  // namespace kaldi
