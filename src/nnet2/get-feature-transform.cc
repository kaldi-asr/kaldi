// nnet2/get-feature-transform.cc

// Copyright 2009-2011  Jan Silovsky
//                2013  Johns Hopkins University (author: Daniel Povey)

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


#include "nnet2/get-feature-transform.h"

namespace kaldi {



void FeatureTransformEstimate::Estimate(const FeatureTransformEstimateOptions &opts,
                                        Matrix<BaseFloat> *M,
                                        TpMatrix<BaseFloat> *C) const { 
  double count;
  Vector<double> total_mean;
  SpMatrix<double> total_covar, between_covar;
  GetStats(&total_covar, &between_covar, &total_mean, &count);
  KALDI_LOG << "Data count is " << count;
  EstimateInternal(opts, total_covar, between_covar, total_mean, M, C);
}

// static
void FeatureTransformEstimate::EstimateInternal(
    const FeatureTransformEstimateOptions &opts,
    const SpMatrix<double> &total_covar,
    const SpMatrix<double> &between_covar,
    const Vector<double> &total_mean,
    Matrix<BaseFloat> *M,
    TpMatrix<BaseFloat> *C) {
  
  int32 target_dim = opts.dim, dim = total_covar.NumRows();
  KALDI_ASSERT(target_dim > 0);
  // between-class covar is of most rank C-1
  KALDI_ASSERT(target_dim <= dim);
  
  // within-class covariance
  SpMatrix<double> wc_covar(total_covar);
  wc_covar.AddSp(-1.0, between_covar);
  TpMatrix<double> wc_covar_sqrt(dim);
  try {
    wc_covar_sqrt.Cholesky(wc_covar);
    if (C != NULL) {
      C->Resize(dim);
      C->CopyFromTp(wc_covar_sqrt);
    }
  } catch (...) {
    BaseFloat smooth = 1.0e-03 * wc_covar.Trace() / wc_covar.NumRows();
    KALDI_LOG << "Cholesky failed (possibly not +ve definite), so adding " << smooth
              << " to diagonal and trying again.\n";
    for (int32 i = 0; i < wc_covar.NumRows(); i++)
      wc_covar(i, i) += smooth;
    wc_covar_sqrt.Cholesky(wc_covar);    
  }
  Matrix<double> wc_covar_sqrt_mat(wc_covar_sqrt);
  wc_covar_sqrt_mat.Invert();

  SpMatrix<double> tmp_sp(dim);
  tmp_sp.AddMat2Sp(1.0, wc_covar_sqrt_mat, kNoTrans, between_covar, 0.0);
  Matrix<double> tmp_mat(tmp_sp);
  Matrix<double> svd_u(dim, dim), svd_vt(dim, dim);
  Vector<double> svd_d(dim);
  tmp_mat.Svd(&svd_d, &svd_u, &svd_vt);
  SortSvd(&svd_d, &svd_u);

  KALDI_LOG << "LDA singular values are " << svd_d;

  KALDI_LOG << "Sum of all singular values is " << svd_d.Sum();
  KALDI_LOG << "Sum of selected singular values is " <<
      SubVector<double>(svd_d, 0, target_dim).Sum();
  
  Matrix<double> lda_mat(dim, dim);
  lda_mat.AddMatMat(1.0, svd_u, kTrans, wc_covar_sqrt_mat, kNoTrans, 0.0);

  // finally, copy first target_dim rows to m
  M->Resize(target_dim, dim);
  M->CopyFromMat(lda_mat.Range(0, target_dim, 0, dim));
  
  if (opts.within_class_factor != 1.0) {
    for (int32 i = 0; i < svd_d.Dim(); i++) {
      BaseFloat old_var = 1.0 + svd_d(i), // the total variance of that dim..
          new_var = opts.within_class_factor + svd_d(i), // the variance we want..
          scale = sqrt(new_var / old_var);
      if (i < M->NumRows())
        M->Row(i).Scale(scale);
    }
  }

  if (opts.max_singular_value > 0.0) {
    int32 rows = M->NumRows(), cols = M->NumCols(),
        min_dim = std::min(rows, cols);
    Matrix<BaseFloat> U(rows, min_dim), Vt(min_dim, cols);
    Vector<BaseFloat> s(min_dim);
    M->Svd(&s, &U, &Vt); // decompose m = U diag(s) Vt.
    BaseFloat max_s = s.Max();
    int32 n = s.ApplyCeiling(opts.max_singular_value);
    if (n > 0) {
      KALDI_LOG << "Applied ceiling to " << n << " out of " << s.Dim()
                << " singular values of transform using ceiling "
                << opts.max_singular_value << ", max is " << max_s;
      Vt.MulRowsVec(s);
      // reconstruct m with the modified singular values:
      M->AddMatMat(1.0, U, kNoTrans, Vt, kNoTrans, 0.0);
    }
  }

  if (opts.remove_offset)
    AddMeanOffset(total_mean, M);
}

void FeatureTransformEstimateMulti::EstimateTransformPart(
    const FeatureTransformEstimateOptions &opts,
    const std::vector<int32> &indexes,
    const SpMatrix<double> &total_covar,
    const SpMatrix<double> &between_covar,
    const Vector<double> &mean,
    Matrix<BaseFloat> *M) const {

  int32 full_dim = Dim(), proj_dim = indexes.size();
  Matrix<double> transform(proj_dim, full_dim); // projects from full to projected dim.
  for (int32 i = 0; i < proj_dim; i++)
    transform(i, indexes[i]) = 1.0;

  SpMatrix<double> total_covar_proj(proj_dim), between_covar_proj(proj_dim);
  Vector<double> mean_proj(proj_dim);
  total_covar_proj.AddMat2Sp(1.0, transform, kNoTrans, total_covar, 0.0);
  between_covar_proj.AddMat2Sp(1.0, transform, kNoTrans, between_covar, 0.0);
  mean_proj.AddMatVec(1.0, transform, kNoTrans, mean, 0.0);

  Matrix<BaseFloat> M_proj;
  FeatureTransformEstimateOptions opts_tmp(opts);
  opts_tmp.dim = proj_dim;
  EstimateInternal(opts_tmp, total_covar_proj, between_covar_proj, mean_proj,
                   &M_proj, NULL);
  if (M_proj.NumCols() == proj_dim + 1) { // Extend transform to add the extra "1" that we
                                          // use to handle mean shifts..
    transform.Resize(proj_dim + 1, full_dim + 1, kCopyData);
    transform(proj_dim, full_dim) = 1.0;
  }
  M->Resize(proj_dim, transform.NumCols());
  // Produce output..
  M->AddMatMat(1.0, M_proj, kNoTrans, Matrix<BaseFloat>(transform),
               kNoTrans, 0.0);
}

void FeatureTransformEstimateMulti::Estimate(
    const FeatureTransformEstimateOptions &opts,
    const std::vector<std::vector<int32> > &indexes,
    Matrix<BaseFloat> *M) const {

  int32 input_dim = Dim(), output_dim = 0, num_transforms = indexes.size();
  for (int32 i = 0; i < num_transforms; i++) { // some input-checking.
    KALDI_ASSERT(indexes[i].size() > 0);
    std::vector<int32> this_indexes(indexes[i]);
    std::sort(this_indexes.begin(), this_indexes.end());
    KALDI_ASSERT(IsSortedAndUniq(this_indexes)); // check for duplicates.
    KALDI_ASSERT(this_indexes.front() >= 0);
    KALDI_ASSERT(this_indexes.back() < input_dim);
    output_dim += this_indexes.size();
  }

  int32 input_dim_ext = (opts.remove_offset ? input_dim + 1 : input_dim);
  M->Resize(output_dim, input_dim_ext);
  
  double count;
  Vector<double> total_mean;
  SpMatrix<double> total_covar, between_covar;
  GetStats(&total_covar, &between_covar, &total_mean, &count);

  int32 cur_output_index = 0;
  for (int32 i = 0; i < num_transforms; i++) {
    Matrix<BaseFloat> M_tmp;
    EstimateTransformPart(opts, indexes[i], total_covar, between_covar,
                          total_mean, &M_tmp);
    int32 this_output_dim = indexes[i].size();
    M->Range(cur_output_index, this_output_dim, 0, M->NumCols()).
        CopyFromMat(M_tmp);
    cur_output_index += this_output_dim;
  }
  
}


}  // End of namespace kaldi
