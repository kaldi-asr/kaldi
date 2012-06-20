// transform/basis-fmllr-diag-gmm.h

// Copyright 2009-2011  Carnegie Mellon University; Johns Hopkins University;
//                      Yajie Miao  Dan Povey

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


#ifndef KALDI_BASIS_FMLLR_DIAG_GMM_H_
#define KALDI_BASIS_FMLLR_DIAG_GMM_H_

#include <vector>
#include <string>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/mle-full-gmm.h"
#include "gmm/mle-am-diag-gmm.h"
#include "transform/transform-common.h"
#include "util/kaldi-table.h"
#include "util/kaldi-holder.h"

namespace kaldi {

/* This header contains routines for performing subspace CMLLR
   (without a regression tree) for diagonal GMM acoustic model.

   Refer to Dan Povey's paper for derivations:
   Daniel Povey, Kaisheng Yao. A basis representation of constrained
   MLLR transforms for robust adaptation. Computer Speech and Language,
   volume 26:35–51, 2012.
*/

struct BasisFmllrOptions {
  int32 num_iters;
  BaseFloat size_scale;
  BaseFloat min_count;
  int32 step_size_iters;
  BasisFmllrOptions(): num_iters(10), size_scale(0.2), min_count(500.0), step_size_iters(3) { }
  void Register(ParseOptions *po) {
    po->Register("num-iters", &num_iters,
                 "Number of iterations in basis fMLLR update during testing");
    po->Register("size-scale", &size_scale,
                 "Scale (< 1.0) of speaker occupancy to decide base number");
    po->Register("fmllr-min-count", &min_count,
                 "Minimum count required to update fMLLR");
    po->Register("step-size-iters", &step_size_iters,
                 "Number of iterations in computing step size");
  }
};


/** \class BasisFmllrGlobalParams
 *  Global parameters for basis fMLLR.
 */
class BasisFmllrGlobalParams {

 public:
  BasisFmllrGlobalParams() { }
  explicit BasisFmllrGlobalParams(int32 dim) {
	  dim_ = dim;
	  ResizeParams(dim);
  }

  void ResizeParams(int32 dim);

  /// Routines for reading and writing global parameters
  void Write(std::ostream &out_stream, bool binary) const;
  void Read(std::istream &in_stream, bool binary, bool add = false);

  /// Accumulate gradient scatter for one (training) speaker.
  /// To finish the process, we need to traverse the whole training
  /// set. Parallelization works if the speaker list is splitted, and
  /// global parameters are summed up by setting add=true in ReadParams
  /// See section 5.2 of the paper.
  void AccuGradientScatter(const AffineXformStats &spk_stats);

  /// Estimate the base matrices efficiently in a Maximum Likelihood manner.
  /// It takes diagonal GMM as argument, which will be used for preconditioner
  /// computation. This function returns the total number of bases, which is
  /// fixed as
  /// N = (dim + 1) * dim
  /// Note that SVD is performed in the normalized space. The base matrices
  /// are finally converted back to the unnormalized space.
  void EstimateFmllrBasis(const AmDiagGmm &am_gmm,
  		                  int32* base_num = NULL);

  /// Gradient scatter. Dim is [(D+1)*D] [(D+1)*D]
  SpMatrix<BaseFloat> grad_scatter_;
  /// Basis matrice. Dim is [T] [D] [D+1]
  /// T is the number of bases
  vector< Matrix<BaseFloat> > fmllr_basis_;
  /// Feature dimension
  int32 dim_;
  /// Number of bases D*(D+1)
  int32 basis_size_;

};

/// This function computes the preconditioner matrix, prior to gradient
/// scatter accumulation. Since the expected values of G statistics are
/// used, it takes the acoustic model as the argument, rather than the
/// actual accumulations AffineXformStats
/// See section 5.1 of the paper.
void ComputeAmDiagPrecond(const AmDiagGmm &am_gmm,
                          SpMatrix<double>* pre_cond);

/// This function performs speaker adaptation, computing the fMLLR matrix
/// based on speaker statistics. It takes the global params (i.e., base matrices)
/// as argument. The basis weights (d_{1}, d_{2}, ..., d_{N}) are optimized
/// implicitly. Optionally, it can also return the weights explicitly, if the
/// argument \coefficient is initialized as not NULL.
/// See section 5.3 of the paper for more details.
void BasisFmllrCoefficients(const BasisFmllrGlobalParams &basis_params,
		                    std::string speaker,   // just for logging and debugging
		                    const AffineXformStats &spk_stats,
	                        Matrix<BaseFloat>* out_xform,
	                        Vector<BaseFloat>* coefficient,
	                        BaseFloat* objf_impr,
	                        BaseFloat* count,
	                        BasisFmllrOptions options);

/// This function takes the step direction (delta) of fMLLR matrix as argument,
/// and optimize step size using Newton's method. This is an iterative method,
/// where each iteration should not decrease the auxiliary function. Note that
/// the resulting step size \k should be close to 1. If \k <<1 or >>1, there
/// maybe problems with preconditioning or the speaker stats.
double CalBasisFmllrStepSize(const AffineXformStats &spk_stats,
                             const Matrix<double> &delta,
                             const Matrix<double> &A,
                             const Matrix<double> &S,
                             int32 max_iters);



} // namespace kaldi

#endif  // KALDI_BASIS_FMLLR_DIAG_GMM_H_
