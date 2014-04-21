// transform/mllt.cc

// Copyright 2009-2011 Microsoft Corporation

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

#include "transform/mllt.h"
#include "util/const-integer-set.h"

namespace kaldi {

void MlltAccs::Init(int32 dim, BaseFloat rand_prune) {  // initializes (destroys anything that was there before).
  KALDI_ASSERT(dim > 0);
  beta_ = 0;
  rand_prune_ = rand_prune;
  G_.resize(dim);
  for (int32 i = 0; i < dim; i++)
    G_[i].Resize(dim);  // will zero it too.
}

void MlltAccs::Read(std::istream &is, bool binary, bool add) {
  ExpectToken(is, binary, "<MlltAccs>");
  double beta;
  int32 dim;
  ReadBasicType(is, binary, &beta);
  if (!add) beta_ = beta;
  else beta_ += beta;
  ReadBasicType(is, binary, &dim);
  if (add && G_.size() != 0 && static_cast<size_t>(dim) != G_.size())
    KALDI_ERR << "MlltAccs::Read, summing accs of different size.";
  if (!add || G_.empty()) G_.resize(dim);
  ExpectToken(is, binary, "<G>");
  for (size_t i = 0; i < G_.size(); i++)
    G_[i].Read(is, binary, add);
  ExpectToken(is, binary, "</MlltAccs>");
}

void MlltAccs::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<MlltAccs>");
  if(!binary) os << '\n';
  WriteBasicType(os, binary, beta_);
  int32 dim = G_.size();
  WriteBasicType(os, binary, dim);
  WriteToken(os, binary, "<G>");
  if(!binary) os << '\n';
  for (size_t i = 0; i < G_.size(); i++)
    G_[i].Write(os, binary);
  WriteToken(os, binary, "</MlltAccs>");
  if(!binary) os << '\n';
}

// static version of the Update function.
void MlltAccs::Update(double beta,
                      const std::vector<SpMatrix<double> > &G,
                      MatrixBase<BaseFloat> *M_ptr,
                      BaseFloat *objf_impr_out,
                      BaseFloat *count_out) {
  int32 dim = G.size();
  KALDI_ASSERT(dim != 0 && M_ptr != NULL
               && M_ptr->NumRows() == dim
               && M_ptr->NumCols() == dim);
  if (beta < 10*dim) {  // not really enough data to estimate.
    // don't bother with min-count parameter etc., as MLLT is typically
    // global.
    if (beta > 2*dim)
      KALDI_WARN << "Mllt:Update, very small count " << beta;
    else
      KALDI_WARN << "Mllt:Update, insufficient count " << beta;
  }
  int32 num_iters = 200;  // may later make this an option.
  Matrix<double> M(dim, dim), Minv(dim, dim);
  M.CopyFromMat(*M_ptr);
  std::vector<SpMatrix<double> > Ginv(dim);
  for (int32 i = 0; i < dim;  i++) {
    Ginv[i].Resize(dim);
    Ginv[i].CopyFromSp(G[i]);
    Ginv[i].Invert();
  }

  double tot_objf_impr = 0.0;
  for (int32 p = 0; p < num_iters; p++) {
    for (int32 i = 0; i < dim; i++) {  // for each row
      SubVector<double> row(M, i);
      // work out cofactor (actually cofactor times a constant which
      // doesn't affect anything):
      Minv.CopyFromMat(M);
      Minv.Invert();
      Minv.Transpose();
      SubVector<double> cofactor(Minv, i);
      // Objf is: beta log(|row . cofactor|) -0.5 row^T G[i] row
      // optimized by (c.f. Mark Gales's techreport "semitied covariance matrices
      // for hidden markov models, eq.  (22)),
      // row = G_i^{-1} cofactor sqrt(beta / cofactor^T G_i^{-1} cofactor). (1)
      // here, "row" and "cofactor" are considered as column vectors.
      double objf_before = beta * log(std::abs(VecVec(row, cofactor)))
          -0.5 * VecSpVec(row, G[i], row);
      // do eq. (1) above:
      row.AddSpVec(std::sqrt(beta / VecSpVec(cofactor, Ginv[i], cofactor)),
                   Ginv[i], cofactor, 0.0);
      double objf_after = beta * log(std::abs(VecVec(row, cofactor)))
          -0.5 * VecSpVec(row, G[i], row);
      if (objf_after < objf_before - fabs(objf_before)*0.00001)
        KALDI_ERR << "Objective decrease in MLLT update.";
      tot_objf_impr += objf_after - objf_before;
    }
    if (p < 10 || p % 10 == 0)
      KALDI_LOG << "MLLT objective improvement per frame by " << p
                << "'th iteration is " << (tot_objf_impr/beta) << " per frame "
                << "over " << beta << " frames.";
  }
  if (objf_impr_out)
    *objf_impr_out = tot_objf_impr;
  if (count_out)
    *count_out = beta;
  M_ptr->CopyFromMat(M);
}

void MlltAccs::AccumulateFromPosteriors(const DiagGmm &gmm,
                                        const VectorBase<BaseFloat> &data,
                                        const VectorBase<BaseFloat> &posteriors) {
  KALDI_ASSERT(data.Dim() == gmm.Dim());
  KALDI_ASSERT(data.Dim() == Dim());
  KALDI_ASSERT(posteriors.Dim() == gmm.NumGauss());
  const Matrix<BaseFloat> &means_invvars = gmm.means_invvars();
  const Matrix<BaseFloat> &inv_vars = gmm.inv_vars();
  Vector<BaseFloat> mean(data.Dim());
  SpMatrix<double> tmp(data.Dim());
  Vector<double> offset_dbl(data.Dim());
  double this_beta_ = 0.0;
  KALDI_ASSERT(rand_prune_ >= 0.0);
  for (int32 i = 0; i < posteriors.Dim(); i++) {  // for each mixcomp..
    BaseFloat posterior = RandPrune(posteriors(i), rand_prune_);
    if (posterior == 0.0) continue;
    SubVector<BaseFloat> mean_invvar(means_invvars, i);
    SubVector<BaseFloat> inv_var(inv_vars, i);
    mean.AddVecDivVec(1.0, mean_invvar, inv_var, 0.0);  // get mean.
    mean.AddVec(-1.0, data);  // get offset
    offset_dbl.CopyFromVec(mean);  // make it double.
    tmp.SetZero();
    tmp.AddVec2(1.0, offset_dbl);
    for (int32 j = 0; j < data.Dim(); j++)
      G_[j].AddSp(inv_var(j)*posterior, tmp);
    this_beta_ += posterior;
  }
  beta_ += this_beta_;
  Vector<double> data_dbl(data);
}

BaseFloat MlltAccs::AccumulateFromGmm(const DiagGmm &gmm,
                                      const VectorBase<BaseFloat> &data,
                                      BaseFloat weight) {  // e.g. weight = 1.0
  Vector<BaseFloat> posteriors(gmm.NumGauss());
  BaseFloat ans = gmm.ComponentPosteriors(data, &posteriors);
  posteriors.Scale(weight);
  AccumulateFromPosteriors(gmm, data, posteriors);
  return ans;
}


BaseFloat MlltAccs::AccumulateFromGmmPreselect(
    const DiagGmm &gmm,
    const std::vector<int32> &gselect,
    const VectorBase<BaseFloat> &data,
    BaseFloat weight) {  // e.g. weight = 1.0
  KALDI_ASSERT(!gselect.empty());
  Vector<BaseFloat> loglikes;
  gmm.LogLikelihoodsPreselect(data, gselect, &loglikes);
  BaseFloat loglike = loglikes.ApplySoftMax();
  // now "loglikes" is a vector of posteriors, indexed
  // by the same index as gselect.
  Vector<BaseFloat> posteriors(gmm.NumGauss());
  for (size_t i = 0; i < gselect.size(); i++)
    posteriors(gselect[i]) = loglikes(i) * weight;
  AccumulateFromPosteriors(gmm, data, posteriors);
  return loglike;
}




} // namespace kaldi
