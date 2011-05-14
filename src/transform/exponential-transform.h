// transform/exponential-transform.h

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

#ifndef KALDI_TRANSFORM_EXPONENTIAL_TRANSFORM_H_
#define KALDI_TRANSFORM_EXPONENTIAL_TRANSFORM_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "transform/fmllr-diag-gmm.h"

namespace kaldi {

// We define an exponential transform is a transform of the form
// W_s = D_s exp(t_s A) B, which takes x^+ -> x (where ^+ is adding a one);
// only t_s and D_s are speaker-specific.  t roughly represents the vtln warp
// factor (although it can be positive or negative and is continuous), and
// D_s is either a diagonal or an offset-only fMLLR matrix (or just
// [ I ; 0 ], so no mean even), depending on options.
// "exp" here is matrix exponential, defined by exp(A) = I + A + 1/2! A A + 1/3! A A A + ...
// note that the last row of A is 0 0 0 ...  and the last row of B is
// 0 0 0 ... 0 1.  The "globally trained" things are A and B.
// We train A and B on separate iterations.


enum EtNormalizeType {
  kEtNormalizeMean,
  kEtNormalizeMeanAndVar,
  kEtNormalizeNone
};

class ExponentialTransformAccsA;
class ExponentialTransformAccsB;


// Class ExponentialTransform holds just the globally shared parts of the exponential
// transform, i.e. A_ and B_.
class ExponentialTransform {
 public:
  ExponentialTransform() { } // typically use this constructor only prior to
  // calling Read().

  ExponentialTransform(int32 dim, EtNormalizeType norm_type, int32 seed = 0) {
    Init(dim, norm_type, seed);
  }

  void Init(int32 dim,
            EtNormalizeType norm_type,
            int32 seed = 0);  // Initializes A to a pseudo-random unit-norm matrix
  // (with last row zero), and B to unity.  "dim" is the feature dim, so both A and B
  // are of dimension dim+1

  // SetNormalizeType sets the normalization type to this.  But it only allows
  // you to increase the normalization type, i.e. None->Mean or MeanAndVar,
  // or Mean->MeanAndVar.
  void SetNormalizeType(EtNormalizeType norm_type);

  // ComputeTransform does not attempt to work out the objective function change,
  // because of possible confusion about what the correct baseline should be.
  // You can use FmllrAuxFuncDiagGmm to measure the change.
  void ComputeTransform(const FmllrDiagGmmAccs &accs,
                        MatrixBase<BaseFloat> *Ws,  // output fMLLR transform, should be size dim x dim+1
                        BaseFloat *t,
                        MatrixBase<BaseFloat> *Ds,
                        BaseFloat *objf_impr = NULL,  // versus just B
                        BaseFloat *count = NULL);

  int32 Dim() const { return A_.NumRows() - 1; }  // returns feature dim.

  // Ds is the first term in
  // fmllr_mat = W_s = D_s exp(t_s A) B, which is a diagonal-only fMLLR (or possibly
  // just mean-offset or [ I; 0 ], depending on whether norm_type_ is
  // {MeanAndVar, Mean, None}.

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  /// Returns B minus its last row, which is the closest thing to a "default transform"
  /// that we have.
  void GetDefaultTransform(Matrix<BaseFloat> *transform) const;


  void ComputeDs(const MatrixBase<BaseFloat> &Ws,
                 BaseFloat t,
                 MatrixBase<BaseFloat> *Ds) const;  // Computes the D_s matrix,
  // given W_s and  the value of t.

  friend class ExponentialTransformAccsA;
  friend class ExponentialTransformAccsB;
 protected:
  Matrix<BaseFloat> A_;  // d+1 by d+1 matrix; last row 0 0 0 .. 0 0.
  Matrix<BaseFloat> B_;  // d+1 by d+1 matrix; last row 0 0 0 .. 0 1.
  EtNormalizeType norm_type_;  // tells us how to train D_s.
 private:
  static void ComposeAffineTransforms(const MatrixBase<BaseFloat> &A,
                                      const MatrixBase<BaseFloat> &B,
                                      MatrixBase<BaseFloat> *C);


};

// This is an MLLT type of update.
class ExponentialTransformAccsB {
 public:
  ExponentialTransformAccsB() { } // typically only used prior to Read().

  ExponentialTransformAccsB(int32 dim) { Init(dim); }

  void Init(int32 dim);

  // AccumulateFromPosteriors is as in the base class, except we
  // supply the transform D_s (expected to be a diagonal or mean-only
  // transform), which is treated as a model-space transform here.
  // Here, "t_data" is the data transformed by the transform W_s.
  // Be careful-- this is different from the accumulation for A, in which
  // the fMLLR stats are accumulated given the original data.
  void AccumulateFromPosteriors(const DiagGmm &gmm,
                                const VectorBase<BaseFloat> &t_data,
                                const VectorBase<BaseFloat> &posteriors,
                                const MatrixBase<BaseFloat> &Ds);



  // The Update function does the MLLT update for B.  It sets "M"
  // to the transform that we would have to apply to the model means.
  void Update(ExponentialTransform *et,
              BaseFloat *objf_impr,
              BaseFloat *count,
              MatrixBase<BaseFloat> *M);

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary, bool add = false);

 private:
  double beta_;
  std::vector<SpMatrix<double> > G_;

};



struct ExponentialTransformUpdateAOptions {
  BaseFloat learning_rate;
  bool renormalize;  // renormalize A and recenter the warp factors on each iteration...
  ExponentialTransformUpdateAOptions(): learning_rate(1.0), renormalize(true) { }
  void Register(ParseOptions *po) {
    po->Register("learning-rate", &learning_rate, "Learning rate for updating A (make <1 if instability suspected)\n");
    po->Register("renormalize", &renormalize, "True if you want to renormalize the warp factors on each iteration of update (recommended).");
  }
};


class ExponentialTransformAccsA {
 public:
  // This class does the accumulation and upate for the "A" part of the

  // global transform.
  // AccumulateForSpeaker does the accumulation for the speaker,
  // given standard fMLLR accs that have been accumulated given the
  // un-transformed data.
  void AccumulateForSpeaker(const FmllrDiagGmmAccs &accs,
                            const ExponentialTransform &et,
                            const MatrixBase<BaseFloat> &Ds,
                            BaseFloat t);

  ExponentialTransformAccsA() { } // typically use this constructor prior to Read().

  ExponentialTransformAccsA(int32 dim) { Init(dim); }

  void Init(int32 dim);

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary, bool add = false);

  // Updates the matrix A (also changes B as a side effect).
  void Update(const ExponentialTransformUpdateAOptions &opts,
              ExponentialTransform *et,
              BaseFloat *objf_impr,
              BaseFloat *count);

 private:
  double beta_;  // sum of speaker betas.  for diagnostics.
  double beta_t_;  // sum of speaker betas times T.  for log-det term.
  std::vector<SpMatrix<double> > G_;  // Like the G stats of
  // fMLLR, taken after the B transform.  Summed over speakers and
  // weighted by t^2.
  Matrix<double> F_;  // local gradient w.r.t. the first d rows of A.
};





}  // End namespace kaldi

#endif  // KALDI_TRANSFORM_LDA_ESTIMATE_H_

