// transform/lda-estimate.cc

// Copyright 2009-2011  Jan Silovsky
//                2013  Johns Hopkins University

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


#include "transform/lda-estimate.h"

namespace kaldi {

void LdaEstimate::Init(int32 num_classes, int32 dimension) {
  zero_acc_.Resize(num_classes);
  first_acc_.Resize(num_classes, dimension);
  total_second_acc_.Resize(dimension);
}

void LdaEstimate::ZeroAccumulators() {
  zero_acc_.SetZero();
  first_acc_.SetZero();
  total_second_acc_.SetZero();
}

void LdaEstimate::Scale(BaseFloat f) {
  double d = static_cast<double>(f);
  zero_acc_.Scale(d);
  first_acc_.Scale(d);
  total_second_acc_.Scale(d);
}

void LdaEstimate::Accumulate(const VectorBase<BaseFloat> &data,
                             int32 class_id, BaseFloat weight) {
  KALDI_ASSERT(class_id >= 0);
  KALDI_ASSERT(class_id < NumClasses() && data.Dim() == Dim());

  Vector<double> data_d(data);

  zero_acc_(class_id) += weight;
  first_acc_.Row(class_id).AddVec(weight, data_d);
  total_second_acc_.AddVec2(weight, data_d);
}

void LdaEstimate::GetStats(SpMatrix<double> *total_covar,
                           SpMatrix<double> *between_covar,
                           Vector<double> *total_mean,
                           double *tot_count) const {
  int32 num_class = NumClasses(), dim = Dim();
  double sum = zero_acc_.Sum();
  *tot_count = sum;
  total_covar->Resize(dim);
  total_covar->CopyFromSp(total_second_acc_);
  total_mean->Resize(dim);
  total_mean->AddRowSumMat(1.0, first_acc_);
  total_mean->Scale(1.0 / sum);
  total_covar->Scale(1.0 / sum);
  total_covar->AddVec2(-1.0, *total_mean);
  
  between_covar->Resize(dim);
  Vector<double> class_mean(dim);
  for (int32 c = 0; c < num_class; c++) {
    if (zero_acc_(c) != 0.0) {
      class_mean.CopyRowFromMat(first_acc_, c);
      class_mean.Scale(1.0 / zero_acc_(c));
      between_covar->AddVec2(zero_acc_(c) / sum, class_mean);
    }
  }
  between_covar->AddVec2(-1.0, *total_mean);
}


void LdaEstimate::Estimate(const LdaEstimateOptions &opts,
                           Matrix<BaseFloat> *m,
                           Matrix<BaseFloat> *mfull) const {
  int32 target_dim = opts.dim;
  KALDI_ASSERT(target_dim > 0);
  // between-class covar is of most rank C-1
  KALDI_ASSERT(target_dim <= Dim() && (target_dim < NumClasses() || opts.allow_large_dim));
  int32 dim = Dim();
  
  double count;
  SpMatrix<double> total_covar, bc_covar;
  Vector<double> total_mean;
  GetStats(&total_covar, &bc_covar, &total_mean, &count);
  
  // within-class covariance
  SpMatrix<double> wc_covar(total_covar);
  wc_covar.AddSp(-1.0, bc_covar);
  TpMatrix<double> wc_covar_sqrt(dim);
  try {
    wc_covar_sqrt.Cholesky(wc_covar);
  } catch (...) {
    BaseFloat smooth = 1.0e-03 * wc_covar.Trace() / wc_covar.NumRows();
    KALDI_LOG << "Cholesky failed (possibly not +ve definite), so adding " << smooth
              << " to diagonal and trying again.\n";
    for (int32 i = 0; i < wc_covar.NumRows(); i++)
      wc_covar(i, i) += smooth;
    wc_covar_sqrt.Cholesky(wc_covar);    
  }
  Matrix<double> wc_covar_sqrt_mat(wc_covar_sqrt);
  // copy wc_covar_sqrt to Matrix, because it facilitates further use
  wc_covar_sqrt_mat.Invert();

  SpMatrix<double> tmp_sp(dim);
  tmp_sp.AddMat2Sp(1.0, wc_covar_sqrt_mat, kNoTrans, bc_covar, 0.0);
  Matrix<double> tmp_mat(tmp_sp);

  Matrix<double> svd_u(dim, dim), svd_vt(dim, dim);
  Vector<double> svd_d(dim);
  tmp_mat.Svd(&svd_d, &svd_u, &svd_vt);
  SortSvd(&svd_d, &svd_u);

  KALDI_LOG << "Data count is " << count;
  KALDI_LOG << "LDA singular values are " << svd_d;

  KALDI_LOG << "Sum of all singular values is " << svd_d.Sum();
  KALDI_LOG << "Sum of selected singular values is " <<
      SubVector<double>(svd_d, 0, target_dim).Sum();
  
  Matrix<double> lda_mat(dim, dim);
  lda_mat.AddMatMat(1.0, svd_u, kTrans, wc_covar_sqrt_mat, kNoTrans, 0.0);

  // finally, copy first target_dim rows to m
  m->Resize(target_dim, dim);
  m->CopyFromMat(lda_mat.Range(0, target_dim, 0, dim));
  
  if (mfull != NULL) {
    mfull->Resize(dim, dim);
    mfull->CopyFromMat(lda_mat);
  }

  if (opts.within_class_factor != 1.0) { // This is not the normal code path;
    // it's intended for use in neural net inputs.
    for (int32 i = 0; i < svd_d.Dim(); i++) {
      BaseFloat old_var = 1.0 + svd_d(i), // the total variance of that dim..
          new_var = opts.within_class_factor + svd_d(i), // the variance we want..
          scale = sqrt(new_var / old_var);
      if (i < m->NumRows())
        m->Row(i).Scale(scale);
      if (mfull != NULL)
        mfull->Row(i).Scale(scale);
    }
  }

  if (opts.remove_offset) {
    AddMeanOffset(total_mean, m);
    if (mfull != NULL)
      AddMeanOffset(total_mean, mfull);
  }  
}

// static
void LdaEstimate::AddMeanOffset(const VectorBase<double> &mean_dbl,
                                Matrix<BaseFloat> *projection) {
  Vector<BaseFloat> mean(mean_dbl);
  Vector<BaseFloat> neg_projected_mean(projection->NumRows());
  // the negative
  neg_projected_mean.AddMatVec(-1.0, *projection, kNoTrans, mean, 0.0);
  projection->Resize(projection->NumRows(),
                     projection->NumCols() + 1,
                     kCopyData);
  projection->CopyColFromVec(neg_projected_mean, projection->NumCols() - 1);
}



void LdaEstimate::Read(std::istream &in_stream, bool binary, bool add) {
  int32 num_classes, dim;
  std::string token;

  ExpectToken(in_stream, binary, "<LDAACCS>");
  ExpectToken(in_stream, binary, "<VECSIZE>");
  ReadBasicType(in_stream, binary, &dim);
  ExpectToken(in_stream, binary, "<NUMCLASSES>");
  ReadBasicType(in_stream, binary, &num_classes);

  if (add) {
    if (NumClasses() != 0 || Dim() != 0) {
      if (num_classes != NumClasses() || dim != Dim()) {
        KALDI_ERR <<"LdaEstimate::Read, dimension or classes count mismatch, "
                  <<(NumClasses()) << ", " <<(Dim()) << ", "
                  << " vs. " <<(num_classes) << ", " << (dim);
      }
    } else {
      Init(num_classes, dim);
    }
  } else {
    Init(num_classes, dim);
  }

  // these are needed for demangling the variances.
  Vector<double> tmp_zero_acc;
  Matrix<double> tmp_first_acc;
  SpMatrix<double> tmp_sec_acc;

  ReadToken(in_stream, binary, &token);
  while (token != "</LDAACCS>") {
    if (token == "<ZERO_ACCS>") {
      tmp_zero_acc.Read(in_stream, binary, false);
      if (!add) zero_acc_.SetZero();
      zero_acc_.AddVec(1.0, tmp_zero_acc);
      // zero_acc_.Read(in_stream, binary, add);
    } else if (token == "<FIRST_ACCS>") {
      tmp_first_acc.Read(in_stream, binary, false);
      if (!add) first_acc_.SetZero();
      first_acc_.AddMat(1.0, tmp_first_acc);
      // first_acc_.Read(in_stream, binary, add);
    } else if (token == "<SECOND_ACCS>") {
      tmp_sec_acc.Read(in_stream, binary, false);
      for (int32 c = 0; c < static_cast<int32>(NumClasses()); c++) {
        if (tmp_zero_acc(c) != 0)
          tmp_sec_acc.AddVec2(1.0 / tmp_zero_acc(c), tmp_first_acc.Row(c));
      }
      if (!add) total_second_acc_.SetZero();
      total_second_acc_.AddSp(1.0, tmp_sec_acc);
      // total_second_acc_.Read(in_stream, binary, add);
    } else {
      KALDI_ERR << "Unexpected token '" << token << "' in file ";
    }
    ReadToken(in_stream, binary, &token);
  }
}

void LdaEstimate::Write(std::ostream &out_stream, bool binary) const {
  WriteToken(out_stream, binary, "<LDAACCS>");
  WriteToken(out_stream, binary, "<VECSIZE>");
  WriteBasicType(out_stream, binary, static_cast<int32>(Dim()));
  WriteToken(out_stream, binary, "<NUMCLASSES>");
  WriteBasicType(out_stream, binary, static_cast<int32>(NumClasses()));

  WriteToken(out_stream, binary, "<ZERO_ACCS>");
  Vector<BaseFloat> zero_acc_bf(zero_acc_);
  zero_acc_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<FIRST_ACCS>");
  Matrix<BaseFloat> first_acc_bf(first_acc_);
  first_acc_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<SECOND_ACCS>");
  SpMatrix<double> tmp_sec_acc(total_second_acc_);
  for (int32 c = 0; c < static_cast<int32>(NumClasses()); c++) {
    if (zero_acc_(c) != 0)
      tmp_sec_acc.AddVec2(-1.0 / zero_acc_(c), first_acc_.Row(c));
  }
  SpMatrix<BaseFloat> tmp_sec_acc_bf(tmp_sec_acc);
  tmp_sec_acc_bf.Write(out_stream, binary);

  WriteToken(out_stream, binary, "</LDAACCS>");
}


}  // End of namespace kaldi
