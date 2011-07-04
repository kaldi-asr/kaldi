// transform/exponential-transform.cc

// Copyright 2009-2011  Microsoft Corporation

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


#include "transform/exponential-transform.h"
#include "util/common-utils.h"

namespace kaldi {

void
ExponentialTransformAccsA::
AccumulateForSpeaker(const FmllrDiagGmmAccs &accs_in,
                     const ExponentialTransform &et,
                     const MatrixBase<BaseFloat> &Ds,
                     BaseFloat t) {
  // This function does some manipulations on the fMLLR accs
  // and adds them to the accs stored in the current class.

  // accs are currently with original, un-transformed features
  // and original model.
  FmllrDiagGmmAccs accs(accs_in);

  // Apply the transform B to the features.
  ApplyFeatureTransformToStats(et.B_, &accs);

  // Get stats like if we had applied the transform Ds as a model-space
  // transform.
  ApplyModelTransformToStats(Ds, &accs);

  // Next stage is to compute derivative w.r.t exp(tA)
  int32 dim = accs.Dim();

  Matrix<double> tA(dim+1, dim+1), exptA(dim+1, dim+1);
  tA.CopyFromMat(et.A_);
  tA.Scale(t);

  MatrixExponential<double> mexp;
  mexp.Compute(tA, &exptA);  // compute exp(tA)

  // Now, with C = exp(tA), compute df/dC.

  Matrix<double> df_dC(dim+1, dim+1);  // last row of this will be zero.
  SubMatrix<double> top_part(df_dC, 0, dim, 0, dim+1);  // part to which we copy K.
  top_part.CopyFromMat(accs.K_);

  for (int32 i = 0; i < dim; i++) {
    SubVector<double> row(exptA, i);
    // (i'th row of df_dC) -= G[i] * (i'th row of [C == exp(tA)])
    df_dC.Row(i).AddSpVec(-1.0, accs.G_[i], row, 1.0);
  }

  Matrix<double> df_dtA(dim+1, dim+1);  // last row of this does not matter.

  // propagate the gradient back through the matrix exponential.
  mexp.Backprop(df_dC, &df_dtA);

  // now we have gradient w.r.t. A.
  // Keep just the part we need.
  SubMatrix<double> df_dtA_top_part(df_dtA, 0, dim, 0, dim+1);

  // Add to the summed gradient w.r.t the top rows of A.
  Ahat_.AddMat(t, df_dtA_top_part);

  // Now update the G_ stats.  We add our (already-transformed)
  // speaker-specific G stats, times t^2.
  for (int32 i = 0; i < dim; i++) {
    G_[i].AddSp(t*t, accs.G_[i]);
    G_[i](i,i) += t*t * accs.K_(i,i);
  }

  // Update the total beta, and the total beta*t (which is needed
  // for a logdet-related term).
  beta_ += accs.beta_;
  beta_t_ += accs.beta_ * t;
}


  // Updates the matrix A (also changes B as a side effect).
void
ExponentialTransformAccsA::Update(const ExponentialTransformUpdateAOptions &opts,
                                  ExponentialTransform *et,
                                  BaseFloat *objf_impr_out,
                                  BaseFloat *count_out) {

  // some checking here.
  int32 dim = Ahat_.NumRows();
  KALDI_ASSERT(dim == Ahat_.NumCols()-1);

  /// First, just a quadratic update.  Report objf impr.
  /// Then renormalization of A (also produces change in B).

  Matrix<double> grad(Ahat_);  // derivative of f w.r.t. 1st dim rows of A.
  // Must add in log-det term.
  Matrix<double> unit(dim, dim+1);
  unit.SetUnit();
  grad.AddMat(beta_t_, unit);  // add \sum_s beta_s t_s times unity--
  // this is the log-det term (since logdet (exp(tA)) is t tr(A)).
  double objf_impr = 0.0;
  for (int32 i = 0; i < dim; i++) {
    // Auxf is grad(i) . delta -0.5 * delta^T G_i delta, where
    // delta is change in row i.
    double objf_before = 0.0;
    SpMatrix<double> Ginv(G_[i]);
    Ginv.Invert();
    Vector<double> delta(dim+1);  // change in i'th row of A.
    // delta <-- learning_rate .G_i^{-1} . grad_i, if grad_i is i'th row of gradient matrix.
    delta.AddSpVec(opts.learning_rate, Ginv, grad.Row(i), 0.0);
    double objf_after = VecVec(delta, grad.Row(i))
        -0.5 * VecSpVec(delta, G_[i], delta);
    objf_impr += objf_after - objf_before;
    // Commit the change.
    Vector<BaseFloat> delta_bf(delta);
    et->A_.Row(i).AddVec(1.0, delta_bf);
  }
  KALDI_ASSERT(objf_impr >= 0.0);
  KALDI_LOG << "Updating matrix A: objf impr is " << (objf_impr/beta_)
            << " per frame over " << beta_ << " frames.";

  if (objf_impr_out) *objf_impr_out = objf_impr;
  if (count_out) *count_out = beta_;

  if (opts.renormalize) {
    // Next renormalize the warp factors so it's "centered"...
    double avg_warp = beta_t_ / beta_;
    KALDI_LOG << "Average warp is " << avg_warp << " (renormalizing to make it zero)";

    // set B <-- exp(avg_warp * A) B, so that we can subtract
    // avg_warp from all the t_s quantities and make the warps average to zero.
    MatrixExponential<BaseFloat> mexp;
    Matrix<BaseFloat> avg_A(et->A_);
    avg_A.Scale(avg_warp);
    Matrix<BaseFloat> exp_avg_A(dim+1, dim+1);
    mexp.Compute(avg_A, &exp_avg_A);
    Matrix<BaseFloat> new_B(dim+1, dim+1);
    // new_B <-- exp(avg_warp . A) * B.
    new_B.AddMatMat(1.0, exp_avg_A, kNoTrans, et->B_, kNoTrans, 0.0);
    et->B_.CopyFromMat(new_B);

    BaseFloat norm = et->A_.FrobeniusNorm();
    if (norm < 1.0e-10)
      KALDI_WARN << "A has very small norm " << norm;
    et->A_.Scale(1.0 / norm);
  }
}

// static
void
ExponentialTransform::ComposeAffineTransforms(const MatrixBase<BaseFloat> &A,
                                              const MatrixBase<BaseFloat> &B,
                                              MatrixBase<BaseFloat> *C_out) {
  // Compute C = A B^+, where ^+ means appending a row 0 0 0 .. 0 1.
  // C may be same memory as A or B.
  int32 dim = A.NumRows();
  KALDI_ASSERT(A.NumCols() == dim+1 && B.NumRows() == dim
               && B.NumCols() == dim+1 && C_out->NumRows() == dim
               && C_out->NumCols() == dim+1);
  Matrix<BaseFloat> C(dim, dim+1);
  SubMatrix<BaseFloat> A_square(A, 0, dim, 0, dim);
  // C = [square part of A] * B
  C.AddMatMat(1.0, A_square, kNoTrans, B, kNoTrans, 0.0);
  for (int32 i = 0; i < dim; i++)
    C(i, dim) += A(i, dim);
  // offset part of C += offset part of A.
  // or think of it as:
  //  C += [last column of A] * e_{dim+1}, where
  //   e_{dim+1} is unit vector in dim+1'th dimension.
  C_out->CopyFromMat(C);
}


void
ExponentialTransform::Init(int32 dim,
                           EtNormalizeType norm_type,
                           int32 seed) {  // Initializes A to a pseudo-random unit-norm matrix
  srand(seed);
  A_.Resize(dim+1, dim+1);
  for (int32 i = 0; i < dim; i++) // last row stays zero.
    for (int32 j = 0; j < dim+1; j++)
      A_(i, j) = RandGauss();
  A_.Scale(1.0 / A_.FrobeniusNorm());

  B_.Resize(dim+1, dim+1);
  B_.SetUnit();

  norm_type_ = norm_type  ;
}

void
ExponentialTransform::
ComputeTransform(const FmllrDiagGmmAccs &accs_in,
                 MatrixBase<BaseFloat> *Ws_out,  // output fMLLR transform, should be size dim x dim+1
                 BaseFloat *t_out,
                 MatrixBase<BaseFloat> *Ds_out,
                 BaseFloat *objf_impr_out,
                 BaseFloat *count_out) {
  // Checking.
  int32 dim = accs_in.Dim();  // feature dim.
  double beta = accs_in.beta_;
  if (beta == 0.0) {
    KALDI_WARN << "ComputeTransform: no data, returning default transform.";
    if (Ws_out) Ws_out->CopyFromMat(B_.Range(0, dim, 0, dim+1));
    if (Ds_out) Ds_out->SetUnit();
    if (t_out) *t_out = 0.0;
    if (objf_impr_out) *objf_impr_out = 0.0;
    if (count_out) *count_out = 0.0;
    return;
  }
  // We can't really have a min-count because we're probably doing mean
  // normalization, and it makes no sense to use un-normalized means if
  // we trained on normalized means.

  Matrix<BaseFloat> Ds(dim, dim+1);
  Ds.SetUnit();  // Initially the "default" transform.
  BaseFloat t = 0.0;

  FmllrDiagGmmAccs accs(accs_in);
  // (1) Apply B as a feature-space transform to the stats.
  ApplyFeatureTransformToStats(B_, &accs);

  BaseFloat tot_objf_impr = 0.0;

  // (2) iteratively:
  //   recompute Ds; recompute t.

  for (int32 iter = 0; iter < 3; iter++) {
    // 3 is more than enough iters...
    // converges very fast.

    if (norm_type_ != kEtNormalizeNone) {
      // (1) Compute an extra part to Ds.
      Matrix<BaseFloat> Ds_new(dim, dim+1);
      Ds_new.SetUnit();

      BaseFloat objf_impr;
      if (norm_type_ == kEtNormalizeMean) {
        objf_impr = ComputeFmllrMatrixDiagGmmOffset(Ds_new,
                                                    accs,
                                                    &Ds_new);
      } else {
        KALDI_ASSERT(norm_type_ == kEtNormalizeMeanAndVar);
        objf_impr = ComputeFmllrMatrixDiagGmmDiagonal(Ds_new,
                                                      accs,
                                                      &Ds_new);
      }
      KALDI_VLOG(2) << "ComputeTransform, iter = " << iter << ", impr from Ds is "
                    << (objf_impr/beta) << " per frame over " << beta
                    << " frames.";
      tot_objf_impr += objf_impr;
      // this roughly does:
      // Ds <-- Ds Ds_new.
      ComposeAffineTransforms(Ds, Ds_new, &Ds);

      // From now, will treat Ds_new as a model-space transform.
      ApplyModelTransformToStats(Ds_new, &accs);
    }

    // Now estimate t.  At this point we can treat t as zero
    // (any existing t has been put into the accs as a feature
    // transformation).
    // Estimation of T: we have a quadratic auxiliary function.
    // Define C == exp(tA), with the i'th row as c_i (viewed as column vector).
    // The auxf is:
    //   C_{1:d}^T K -0.5 \sum_{i = 1}^d c_i^T G_i c_i + beta logdet(C^--)      (1)
    // where C_{1:d} is the first d rows of C;
    // here, the notation ^-- means taking off the last row and column to make
    // it dimension d x d.
    // Defining D == C - I (the delta in C, versus I), we can rewrite this as
    // following using c_i = d_i + e_i (with e_i the unit vector in dimension i):
    //   D_{1:d}^T J -0.5  \sum_{i = 1}^d d_i^T G_i d_i + beta logdet(C^--)     (2)
    // where J = K - S, and
    // s_i = g_{ii},
    // where s_i is the i'th row of S and g_{ii} is the i'th row of G_i.
    // Now, logdet(C^--) is just equal to t tr(A^--), and we can use
    // a 2'nd order Taylor-series approximation to D as:
    //  D \simeq t A + t^2 A A .
    // Keeping only up to 2nd order terms in (2), we have:
    //  t tr(J^T A_{1:d}) + 0.5 t^2 tr(J^T (A A)_{1:d})
    //   -0.5 t^2 \sum_{i = 1}^d a_i^T G_i a_i
    // + beta t tr(A),
    // using that logdet(exp(tA)^--) is tr(tA^--) = t tr(A^--).
    // Making this a quadratic function in t, we have
    // f(t) = a t -0.5 b t^2,
    // where  a = tr(J^T A_{1:d}) + beta tr(A)
    //        b = \sum_{i = 1}^d a_i^T G_i a_i
    //             -tr(J^T (A A)_{1:d})
    //  [note that if b is negative we can legitimately panic].
    // so the solution is: t = a / b.
    // the t we compute is actually a *change* in t (any previously
    // estimated t is already treated as a feature-space transform and
    // has been put into the stats).

    {  // computing change in t.
      Matrix<BaseFloat> S(dim, dim+1);
      for (int32 i = 0; i < dim; i++)
        for (int32 j = 0; j < dim+1; j++)
          S(i, j) = accs.G_[i](i, j);
      Matrix<BaseFloat> Jplus(dim+1, dim+1);  // J with extra row.
      {
        SubMatrix<BaseFloat> J(Jplus, 0, dim, 0, dim+1);
        J.AddMat(-1.0, S);
        Matrix<BaseFloat> K(accs.K_);
        J.AddMat(1.0, K);  // J += K.
        // J is gradient w.r.t. exp(tA), around t = 0.
      }


      BaseFloat a = TraceMatMat(Jplus, A_, kTrans) + beta*(A_.Trace());

      BaseFloat b1 = 0.0, b2 = TraceMatMatMat(Jplus, kTrans, A_, kNoTrans, A_, kNoTrans);
      for (int32 i = 0; i < dim; i++) {
        Vector<double> a_row_i(A_.Row(i));
        b1 += VecSpVec(a_row_i, accs.G_[i], a_row_i);
      }
      if (b2 > 0.8 * b1)  {
        KALDI_WARN << "Unexpected quantities in optimizing t: b2 = " << b2
                   << ", b1 = " << b1;
        b2 = 0.8 * b1;  // at least ensures update has correct sign.
      }
      BaseFloat delta_t = a / (b1 - b2);
      BaseFloat delta_objf_approx = (a*delta_t) - 0.5*delta_t*delta_t*(b1 - b2);

      Matrix<BaseFloat> tA(A_), expA(dim+1, dim+1);
      tA.Scale(delta_t);
      expA.SetUnit();
      SubMatrix<BaseFloat> expA_part(expA, 0, dim, 0, dim+1);
      BaseFloat old_objf = FmllrAuxFuncDiagGmm(expA_part, accs);
      MatrixExponential<BaseFloat> mexp;
      mexp.Compute(tA, &expA);
      BaseFloat new_objf = FmllrAuxFuncDiagGmm(expA_part, accs);

      KALDI_VLOG(2) << "On iteration " << iter << ", delta-t is " << delta_t
                    << ", objf impr is " << (new_objf-old_objf)/beta
                    << " per frame (approx: " << delta_objf_approx/beta << ")"
                    << " over " << beta << " frames.";
      tot_objf_impr += (new_objf - old_objf);

      t += delta_t;
      ApplyFeatureTransformToStats(expA, &accs);  // modify accs so it's as if we
      // already did this transform.
    }
  }

  // Now generate the outputs.
  if (Ws_out) {
    KALDI_ASSERT(Ws_out->NumRows() == dim && Ws_out->NumCols() == dim+1);
    Matrix<BaseFloat> Ws(dim, dim+1);
    Matrix<BaseFloat> tA(A_), exptA(dim+1, dim+1);
    tA.Scale(t);
    MatrixExponential<BaseFloat> mexp;
    mexp.Compute(tA, &exptA);
    Matrix<BaseFloat> exptAB(dim+1, dim+1);
    // exptAB = exp(tA) * B.
    exptAB.AddMatMat(1.0, exptA, kNoTrans, B_, kNoTrans, 0.0);
    // Ws = Ds * exptAB
    Ws.AddMatMat(1.0, Ds, kNoTrans, exptAB, kNoTrans, 0.0);
    Ws_out->CopyFromMat(Ws);
  }
  if (t_out)
    *t_out = t;
  if (Ds_out) {
    KALDI_ASSERT(Ds_out->NumRows() == dim && Ds_out->NumCols() == dim+1);
    Ds_out->CopyFromMat(Ds);
  }
  if (objf_impr_out) *objf_impr_out = tot_objf_impr;
  if (count_out) *count_out = beta;
  KALDI_VLOG(1) << "Computing exponential transform: objf impr (vs. B only) is "
                << (tot_objf_impr/beta) << " per frame over " << beta
                << " frames.";
}

void
ExponentialTransform::ApplyC(const MatrixBase<BaseFloat> &Cpart) {
  int32 dim = A_.NumRows() - 1;
  KALDI_ASSERT(dim > 0 && Cpart.NumRows() == dim && Cpart.NumCols() == dim);
  
  Matrix<BaseFloat> C(dim+1, dim+1);
  C.SetUnit();
  C.Range(0, dim, 0, dim).CopyFromMat(Cpart);
  Matrix<BaseFloat> tmp(dim+1, dim+1);
  tmp.AddMatMat(1.0, C, kNoTrans, B_, kNoTrans, 0.0);
  B_.CopyFromMat(tmp); // B <-- C B
  tmp.AddMatMat(1.0, C, kNoTrans, A_, kNoTrans, 0.0);
  C.Invert();
  A_.AddMatMat(1.0, tmp, kNoTrans, C, kNoTrans, 0.0); // A <-- C A C^{-1}
}  


void ExponentialTransformAccsA::Init(int32 dim) {
  beta_ = 0.0;
  beta_t_ = 0.0;
  G_.resize(dim);
  for (int32 i = 0; i < dim; i++)
    G_[i].Resize(dim+1);
  Ahat_.Resize(dim, dim+1);
}

void ExponentialTransform::Write(std::ostream &os, bool binary) const {
  WriteMarker(os, binary, "<ExponentialTransform>");
  WriteMarker(os, binary, "<A>");
  A_.Write(os, binary);
  WriteMarker(os, binary, "<B>");
  B_.Write(os, binary);
  WriteMarker(os, binary, "<NormType>");
  int32 i = static_cast<int32>(norm_type_);
  WriteBasicType(os, binary, i);
  WriteMarker(os, binary, "</ExponentialTransform>");
}

void ExponentialTransform::Read(std::istream &is, bool binary) {
  ExpectMarker(is, binary, "<ExponentialTransform>");
  ExpectMarker(is, binary, "<A>");
  A_.Read(is, binary);
  ExpectMarker(is, binary, "<B>");
  B_.Read(is, binary);
  ExpectMarker(is, binary, "<NormType>");
  int32 i;
  ReadBasicType(is, binary, &i);
  norm_type_ = static_cast<EtNormalizeType>(i);
  ExpectMarker(is, binary, "</ExponentialTransform>");
}



void ExponentialTransformAccsA::Write(std::ostream &os, bool binary) const {
  WriteMarker(os, binary, "<ExponentialTransformAccsA>");
  WriteMarker(os, binary, "<Beta>");
  WriteBasicType(os, binary, beta_);
  WriteMarker(os, binary, "<BetaT>");
  WriteBasicType(os, binary, beta_t_);
  WriteMarker(os, binary, "<Dim>");
  int32 dim = G_.size();
  WriteBasicType(os, binary, dim);
  WriteMarker(os, binary, "<G>");
  for (int32 i = 0; i < dim; i++)
    G_[i].Write(os, binary);
  WriteMarker(os, binary, "<Ahat>");
  Ahat_.Write(os, binary);
  WriteMarker(os, binary, "</ExponentialTransformAccsA>");
}

void ExponentialTransformAccsA::Read(std::istream &os, bool binary, bool add) {
  if (G_.empty()) add = false;  // don't add to nonexistent stats...
  ExpectMarker(os, binary, "<ExponentialTransformAccsA>");
  ExpectMarker(os, binary, "<Beta>");
  double beta;
  ReadBasicType(os, binary, &beta);
  if (add) beta_ += beta;
  else beta_ = beta;
  ExpectMarker(os, binary, "<BetaT>");
  double beta_t;
  ReadBasicType(os, binary, &beta_t);
  if (add) beta_t_ += beta_t;
  else beta_t_ = beta_t;
  ExpectMarker(os, binary, "<Dim>");
  int32 dim;
  ReadBasicType(os, binary, &dim);
  if (!add) G_.resize(dim);
  else {
    if (static_cast<size_t>(dim) != G_.size())
      KALDI_ERR << "Reading accs for updating B in exponential transform, "
                << "dim mismatch " << dim << " vs. " << G_.size();
  }
  ExpectMarker(os, binary, "<G>");
  for (size_t i = 0; i < G_.size(); i++)
    G_[i].Read(os, binary, add);
  ExpectMarker(os, binary, "<Ahat>"); 
  Ahat_.Read(os, binary, add);
  ExpectMarker(os, binary, "</ExponentialTransformAccsA>");
}


void ExponentialTransform::SetNormalizeType(EtNormalizeType norm_type) {
  if ((norm_type_ == kEtNormalizeMeanAndVar && norm_type != kEtNormalizeMeanAndVar)
     ||( norm_type_ == kEtNormalizeMean && norm_type == kEtNormalizeNone))
    KALDI_ERR << "SetNormalizeType: trying to reduce the amount of normalization "
              << "(may not be consistent with transform estimation). ";
  norm_type_ = norm_type;
}

void ExponentialTransform::ComputeDs(const MatrixBase<BaseFloat> &Ws,
                                     BaseFloat t,
                                     MatrixBase<BaseFloat> *Ds) const {
  int32 dim = A_.NumRows() - 1;
  Matrix<BaseFloat> mtA(dim+1, dim+1), expmtA(dim+1, dim+1), invB(dim+1, dim+1),
      tmp(dim, dim+1);
  mtA.CopyFromMat(A_);
  mtA.Scale(-t);
  MatrixExponential<BaseFloat> mexp;
  mexp.Compute(mtA, &expmtA);
  invB.CopyFromMat(B_);
  invB.Invert();
  // tmp <-- Ws * B^{-1}.  Note that Ws = Ds . exp(tA) . B,
  // and we are computing Ds = Ws . B^{-1} . exp(-tA).
  tmp.AddMatMat(1.0, Ws, kNoTrans, invB, kNoTrans, 0.0);
  // Ds <-- tmp * exp(-tA)
  (*Ds).AddMatMat(1.0, tmp, kNoTrans, expmtA, kNoTrans, 0.0);
  SubMatrix<BaseFloat> Ds_diag(*Ds, 0, dim, 0, dim);
  if (!Ds_diag.IsDiagonal(0.01)) {
    KALDI_WARN << "ComputeDs: computed D_s is not diagonal, continuing "
        "but this is a serious error (possibly the transforms/warp factors are "
        "incompatible with each other or were computed with a different "
        "exponential-transform object.";
  }
}


void ExponentialTransform::GetDefaultTransform(Matrix<BaseFloat> *transform) const {
  KALDI_ASSERT(transform != NULL);
  KALDI_ASSERT(B_.NumRows() != 0 && B_.NumRows() == B_.NumCols());
  transform->Resize(B_.NumRows() - 1, B_.NumRows());
  // copy all but last row of B, to "transform"
  transform->CopyFromMat(B_.Range(0, B_.NumRows()-1, 0, B_.NumCols()));
}


}  // End of namespace kaldi
