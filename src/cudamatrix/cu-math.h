// cudamatrix/cu-math.h

// Copyright 2009-2012  Karel Vesely

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



#ifndef KALDI_CUDAMATRIX_CUMATH_H_
#define KALDI_CUDAMATRIX_CUMATH_H_

#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-stlvector.h"
#include "cudamatrix/cu-device.h"

#include "util/timer.h"

namespace kaldi {
  
/**
 * Hide the CUDA kernel ANSI-C wrappers to subnamespace cu::
 */
namespace cu {

  /// Logistic sigmoid
  /// Y = Sigmoid(X) : y = 1/(1+exp(-x))
  template<typename Real>
  void Sigmoid(const CuMatrix<Real>& X, CuMatrix<Real>* Y);

  /// Derivative of Logistic sigmoid
  /// Eout = Y(1-Y) .* Ein
  template<typename Real>
  void DiffSigmoid(const CuMatrix<Real>& Ein, const CuMatrix<Real>& Y, CuMatrix<Real>* Eout);

  /// Softmax nonlinearity
  /// Y = Softmax(X) : Yij = e^Xij / sum_k(e^Xik)
  /// for each row, the max value is first subtracted for good numerical stability
  template<typename Real>
  void Softmax(const CuMatrix<Real>& X, CuMatrix<Real>* Y);

  /// apply the L1 regularization
  template<typename Real>
  void RegularizeL1(CuMatrix<Real> *wei, CuMatrix<Real> *grad, Real l1, Real lr);

  /// Find the id of the maximal element for each row
  template<typename Real>
  void FindRowMaxId(const CuMatrix<Real> &mat, CuStlVector<int32> *id);

  /// Differentiate the block [softmax+cross-entropy] :
  /// dE/da = posterior_mat - target_mat, 
  /// 'E' is error function, 'a' is activation on softmax input
  ///
  /// Interface:
  /// tgt ... index vector, encodes the matrix of targets
  /// net_out_or_diff ... before invocation net output, after diff dE/da
  /// log_post_tgt ... per-frame statistics for cross-entropy computations :
  ///                  log(sum_row(posterior_mat .* target_mat))
  template<typename Real>
  void DiffXent(const CuStlVector<int32> &tgt, CuMatrix<Real> *net_out_or_diff, CuVector<Real> *log_post_tgt);

  /// ie. switch rows according to copy_from_idx
  template<typename Real>
  void Randomize(const CuMatrix<Real> &src, const CuStlVector<int32> &copy_from_idx, CuMatrix<Real> *tgt);

  /// ie. concatenate the frames with offsets from frame_offsets
  template<typename Real>
  void Expand(const CuMatrix<Real> &src, const CuStlVector<int32> &frame_offsets, CuMatrix<Real> *tgt);

  /// ie. concatenate the frames with offsets from frame_offsets
  template<typename Real>
  void Copy(const CuMatrix<Real> &src, const CuStlVector<int32> &copy_from_indices, CuMatrix<Real> *tgt);



} // namespace cu
} // namespace kaldi


#include "cu-math-inl.h"

#endif
