// sgmm/estimate-am-sgmm-compress.h

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

#ifndef KALDI_SGMM_ESTIMATE_AM_SGMM_COMPRESS_H_
#define KALDI_SGMM_ESTIMATE_AM_SGMM_COMPRESS_H_

#include <vector>
#include <queue>

#include "sgmm/estimate-am-sgmm.h"

namespace kaldi {

// This class contains some extensions to the basic SGMM update for M and N,
// in which we limit the set of I matrices M_i to a subspace of dimension less
// than I.  We call it CompressM because all the equations are written as if for
// M_i, but it's equally applicable to the N_i (just supply different arguments,
// mapping M->N, Q->R, Y->Z).  We put this in a separate class
// because not part of the original paper on SGMMs and is quite long, and we
// don't want to clutter the main SGMM code.


class SgmmCompressM {
 public:
  SgmmCompressM(const std::vector<Matrix<BaseFloat> > &M,
                const std::vector<Matrix<double> > &Y,
                const std::vector<SpMatrix<BaseFloat> > &SigmaInv,
                const std::vector<SpMatrix<double> > &Q,
                const Vector<double> &gamma,
                int32 G);
  
  void Compute(std::vector<Matrix<BaseFloat> > *M,
               BaseFloat *objf_change_out,  // not normalized by time.
               BaseFloat *tot_t_out);


 private:
  int32 I_, D_, S_, G_;  // I = #Gaussians, D = feature dim, S = subspace dim.
                         // G is dimension we compress to.
  std::vector<Matrix<double> > M_;  // [I][D][S]
  std::vector<SpMatrix<double> > SigmaInv_;  // [I][D], double versions of variances.
  const std::vector<SpMatrix<double> > &Q_;  // [I][S], a reference to Q (outer product
                                           // of vectors), owned outside.
  std::vector<Matrix<double> > J_;  // [I][D][S], the linear statistics: \Sigma_i^{-1} \Y_i
  const Vector<double> &gamma_;  // [i];

  std::vector<Matrix<double> > K_;  // [G][D][S], the basis.
  Matrix<double> a_;  // [I][G], the coefficients.

  int32 num_outer_iters_;
  int32 num_cg_iters_a_;
  int32 num_cg_iters_K_;
  double epsilon_;
  double delta_;

  bool ComputeM();  //  Computes the M from the K and a quantities; returns true
                   // if M was unchanged.

  double Objf() const;  // Returns the objective function we're optimizing, computed
  // from the M, Y, SigmaInv and Q quantities.

  void PcaInit();  // Initializes the K's and a's using PCA; may change M as
  // a result (depending on how the M's were previously obtained).
  void PreconditionForA();  // Preconditioning phase before updating the A's.
  // HessianMulA multiplies by the Hessian when optimizing A.
  void HessianMulA(int32 i, const VectorBase<double> &p, VectorBase<double> *q);
  void ComputeA();
  void PreconditionForKPhase1();  // Phase 1 of preconditioning for K's

  static double KInnerProduct(const std::vector<Matrix<double> > &x,
                              const std::vector<Matrix<double> > &y);

  void ComputeK();

  // multiply by Hessian w.r.t. the set of K's: u <-- A x,
  // where A is the Hessian.
  void HessianMulK(const std::vector<Matrix<double> > &x,
                   std::vector<Matrix<double> > *u) const;

  // Multiply by M^{-1} while estimating K.
  void MulMinv(const std::vector<Matrix<double> > &r,
               const std::vector<SpMatrix<double> > &SigmaG,
               const std::vector<SpMatrix<double> > &QInvG,
               std::vector<Matrix<double> > *z) const;

};


class CompressVars {
 public:
  CompressVars(const Vector<double> &gamma, // counts
               const std::vector<SpMatrix<double> > &T, // variance statistics
                                       // (divide by count to get ML solution)
               int32 G); // subspace dimension

  // The main function.
  void Compute(std::vector<SpMatrix<BaseFloat> > *Sigma,
               BaseFloat *objf_change_out,
               BaseFloat *count_out);

  // Call this if you want to supply (and have computed)
  // the inverse variances.
  void ComputeInv(std::vector<SpMatrix<BaseFloat> > *SigmaInv,
                  BaseFloat *objf_change_out,
                  BaseFloat *count_out);
  
  
 private:

  static void Vectorize(const SpMatrix<double> &S, SubVector<double> *v);
  static void UnVectorize(const SubVector<double> &v, SpMatrix<double> *S);
  
  bool ComputeL();  // Computes the L quantities from a and B.  Returns true
  // if they are unchanged by this.

  void InitS(); // Initialize the C and S quantities (from T and gamma)
  void InitL(const std::vector<SpMatrix<BaseFloat> > &Sigma); // Initialize the L values
  // from Sigma (no PCA, just take log).  Sigmas must be +ve definite.
  
  void InitPca(); // Does the PCA phase of the initialization.

  double Objf() const; // Compute the objective function from the current values of L.
  // This is the actual likelihood per frame times the number of frames, i.e. it contains
  // the factor of -0.5.
  
  void PreconditionForA(); // linearly transform a and B prior to estimating A

  // objf while computing A (as Objf() but for just one i, and no factor of -0.5).
  double ObjfA(int32 i, const VectorBase<double> &a) const;
  
  // objf derivative while computing A. 
  void DerivA(int32 i, const VectorBase<double> &a, VectorBase<double> *d) const;

  // objf while computing B.  No factor of -0.5.
  double ObjfB(const std::vector<SpMatrix<double> > &B) const;

  // Derivative while computing B.  No factorof -0.5.
  void DerivB(const std::vector<SpMatrix<double> > &B,
              std::vector<SpMatrix<double> > *D) const;

  // The inner product we use the CG algorithm while updating B;
  // it returns \sum_g \tr(B_g C_g).
  double InnerProductB(const std::vector<SpMatrix<double> > &B,
                       const std::vector<SpMatrix<double> > &C);

  void CopyB(const std::vector<SpMatrix<double> > &B,
             std::vector<SpMatrix<double> > *C);

  // B <-- B + alpha C.
  void AddB(const std::vector<SpMatrix<double> > &B,
            double alpha,
            std::vector<SpMatrix<double> > *C);
  
  void ScaleB(double alpha, std::vector<SpMatrix<double> > *B);
  
  void ComputeA(); // optimize the coefficients a_{ig}
  void PreconditionForB(); // linearly transform a and B prior to estimating B
  void ComputeB(); // optimize the basis matrices B_g
  void Finalize(std::vector<SpMatrix<BaseFloat> > *Sigma); // compute the variances from the L quantities.
  
  int32 I_, D_, G_; // I = #Gaussians, D = feature dim, G = dimension to

  // compress to.
  const Vector<double> & gamma_;
  const std::vector<SpMatrix<double> > &T_;
  
  Matrix<double> a_; // [I][G], the coefficients a_{ig}
  TpMatrix<double> C_; // Cholesky factor of average variance.
  std::vector<SpMatrix<double> > B_; // [G][D][D], the basis matrices
  std::vector<SpMatrix<double> > L_; // [I][D][D], the current logs of the variance matrices.
  std::vector<SpMatrix<double> > S_; // [I][D][D], the transformed statistics.

  int32 num_outer_iters_;
  int32 num_cg_iters_a_; // num iters when optimizing a.  i_max in math.
  int32 num_cg_iters_B_; // num iters when optimizing B.  i_max in math.
  int32 num_newton_iters_; // j_max in math.
  int32 num_backtrack_iters_; // k_max in math.
  double epsilon_;
  
};  // namespace kaldi



}
#endif  // KALDI_SGMM_ESTIMATE_AM_SGMM_COMPRESS_H_
