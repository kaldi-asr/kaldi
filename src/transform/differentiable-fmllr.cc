// transform/differentiable-fmllr.cc

// Copyright     2018  Johns Hopkins University

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

#include "transform/differentiable-fmllr.h"
#include "matrix/matrix-functions.h"

namespace kaldi {
namespace differentiable_transform {

CoreFmllrEstimator::CoreFmllrEstimator(
    const CoreFmllrEstimatorOptions &opts,
    BaseFloat gamma,
    const MatrixBase<BaseFloat> &G,
    const MatrixBase<BaseFloat> &K,
    MatrixBase<BaseFloat> *A):
    opts_(opts),  gamma_(gamma),
    G_(G), K_(K), A_(A) {
  KALDI_ASSERT(opts.singular_value_relative_floor > 0.0 &&
               gamma > 0.0 && G.NumRows() == K.NumRows() &&
               K.NumRows() == K.NumCols() &&
               SameDim(K, *A));
}


BaseFloat CoreFmllrEstimator::Forward() {
  ComputeH();
  ComputeL();
  ComputeB();
  ComputeA();
  return ComputeObjfChange();
}

void CoreFmllrEstimator::ComputeH() {
  int32 dim = G_.NumRows();
  bool symmetric = true;
  G_rescaler_.Init(&G_, symmetric);
  VectorBase<BaseFloat> &G_singular_values = G_rescaler_.InputSingularValues();
  BaseFloat floor =
      G_singular_values.Max() * opts_.singular_value_relative_floor;
  KALDI_ASSERT(floor > 0.0);
  MatrixIndexT num_floored = 0;
  G_singular_values.ApplyFloor(floor, &num_floored);
  if (num_floored > 0.0)
    KALDI_WARN << num_floored << " out of " << dim
               << " singular values floored in G matrix.";
  VectorBase<BaseFloat>
      &H_singular_values = *G_rescaler_.OutputSingularValues(),
      &H_singular_value_derivs = *G_rescaler_.OutputSingularValueDerivs();
  H_singular_values.CopyFromVec(G_singular_values);
  // H is going to be G^{-0.5}.
  // We don't have to worry about division by zero because we already floored the
  // singular values of G.
  H_singular_values.ApplyPow(-0.5);
  // the derivative of lambda^{-0.5} w.r.t. lambda is -0.5 lambda^{-1.5};
  // we fill in this value in H_singular_value_derivs.
  H_singular_value_derivs.CopyFromVec(G_singular_values);
  H_singular_value_derivs.ApplyPow(-1.5);
  H_singular_value_derivs.Scale(-0.5);
  H_.Resize(dim, dim, kUndefined);
  G_rescaler_.GetOutput(&H_);
}

void CoreFmllrEstimator::ComputeL() {
  int32 dim = G_.NumRows();
  L_.Resize(dim, dim);
  L_.AddMatMat(1.0, K_, kNoTrans, H_, kNoTrans, 0.0);
}

// Compute B = F(L), where F is the
// function that takes the singular values of L, puts them through the function
// f(lamba) = (lambda + sqrt(lambda^2 + 4 gamma)) / 2.
void CoreFmllrEstimator::ComputeB() {
  int32 dim = L_.NumRows();
  bool symmetric = false;
  L_rescaler_.Init(&L_, symmetric);
  VectorBase<BaseFloat> &L_singular_values = L_rescaler_.InputSingularValues();
  BaseFloat floor =
      L_singular_values.Max() * opts_.singular_value_relative_floor;
  KALDI_ASSERT(floor > 0.0);
  MatrixIndexT num_floored = 0;
  L_singular_values.ApplyFloor(floor, &num_floored);
  if (num_floored > 0.0)
    KALDI_WARN << num_floored << " out of " << dim
               << " singular values floored in K matrix.";
  VectorBase<BaseFloat>
      &B_singular_values = *L_rescaler_.OutputSingularValues(),
      &B_singular_value_derivs = *L_rescaler_.OutputSingularValueDerivs();
  // lambda is the original singular value of l,
  // f is where we put f(lambda)
  // f_prime is where we put f'(lambda) (the derivative of f w.r.t lambda).
  BaseFloat *lambda = L_singular_values.Data(),
      *f = B_singular_values.Data(),
      *f_prime = B_singular_value_derivs.Data();

  BaseFloat gamma = gamma_;
  for (int32 i = 0; i < dim; i++) {
    BaseFloat lambda_i = lambda[i];
    f[i] = (lambda_i + std::sqrt(lambda_i * lambda_i + 4.0 * gamma)) / 2.0;
    f_prime[i] = (1.0 + lambda_i /
                  std::sqrt(lambda_i * lambda_i + 4.0 * gamma)) / 2.0;
  }
  B_.Resize(dim, dim, kUndefined);
  L_rescaler_.GetOutput(&B_);
}

void CoreFmllrEstimator::ComputeA() {
  A_->SetZero();  // Make sure there are no NaN's.
  A_->AddMatMat(1.0, B_, kNoTrans, H_, kNoTrans, 0.0);
}

void CoreFmllrEstimator::BackpropA(const MatrixBase<BaseFloat> &A_deriv,
                                   MatrixBase<BaseFloat> *B_deriv,
                                   MatrixBase<BaseFloat> *H_deriv) {
  B_deriv->AddMatMat(1.0, A_deriv, kNoTrans, H_, kTrans, 0.0);
  H_deriv->AddMatMat(1.0, B_, kTrans, A_deriv, kNoTrans, 0.0);
}

void CoreFmllrEstimator::BackpropL(const MatrixBase<BaseFloat> &L_deriv,
                                   MatrixBase<BaseFloat> *K_deriv,
                                   MatrixBase<BaseFloat> *H_deriv) {
  K_deriv->AddMatMat(1.0, L_deriv, kNoTrans, H_, kTrans, 0.0);
  H_deriv->AddMatMat(1.0, K_, kTrans, L_deriv, kNoTrans, 1.0);
}


void CoreFmllrEstimator::Backward(const MatrixBase<BaseFloat> &A_deriv,
                                  Matrix<BaseFloat> *G_deriv,
                                  Matrix<BaseFloat> *K_deriv) {
  KALDI_ASSERT(SameDim(A_deriv, *A_) && SameDim(A_deriv, *G_deriv)
               && SameDim(*G_deriv, *K_deriv));
  int32 dim = A_->NumRows();
  Matrix<BaseFloat> B_deriv(dim, dim), H_deriv(dim, dim),
      L_deriv(dim, dim);
  BackpropA(A_deriv, &B_deriv, &H_deriv);
  // Backprop through the operation B = F(L).
  L_rescaler_.ComputeInputDeriv(B_deriv, &L_deriv);
  BackpropL(L_deriv, K_deriv, &H_deriv);
    // Backprop through the operation H = G^{-0.5}.
  G_rescaler_.ComputeInputDeriv(H_deriv, G_deriv);

  { // Make sure G_deriv is symmetric.  Use H_deriv as a temporary.
    H_deriv.CopyFromMat(*G_deriv);
    G_deriv->AddMat(1.0, H_deriv, kTrans);
    G_deriv->Scale(0.5);
  }
}

}  // namespace differentiable_transform
}  // namespace kaldi
