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
  

  /*
   * float declarations of functions, 
   * the definitions are in cu-math.cc
   */
  /// Logistic sigmoid
  /// Y = Sigmoid(X) : y = 1/(1+exp(-x))
  void Sigmoid(const CuMatrix<float>& X, CuMatrix<float>* Y);

  /// Derivative of Logistic sigmoid
  /// Eout = Y(1-Y) .* Ein
  void DiffSigmoid(const CuMatrix<float>& Ein, const CuMatrix<float>& Y, CuMatrix<float>* Eout);

  /// Softmax nonlinearity
  /// Y = Softmax(X) : Yij = e^Xij / sum_k(e^Xik)
  /// for each row, max value is fist subtracted for numerical stability
  void Softmax(const CuMatrix<float>& X, CuMatrix<float>* Y);

  /// apply the L1 regularization, sets zero to wei and grad elements on zero-crossing
  void RegularizeL1(CuMatrix<float> *wei, CuMatrix<float> *grad, float l1, float lr);

  /// Find the id of the maximal element for each row
  void FindRowMaxId(const CuMatrix<float> &mat, CuStlVector<int32> *id);

  /// Differentiate the block [softmax+cross-entropy]:
  /// dE/da = posterior_mat - target_mat, 
  /// 'E' is error function, 'a' is activation on softmax input
  /// tgt ... index vector, encodes the matrix of targets
  /// net_out_or_diff ... before invocation net output, after diff dE/da
  /// log_post_tgt ... per-frame statistics for cross-entropy computations  
  ///                  : log(sum_row(posterior_mat .* target_mat))
  void DiffXent(const CuStlVector<int32> &tgt, CuMatrix<float> *net_out_or_diff, CuVector<float> *log_post_tgt);

  /// ie. switch rows according to copyFrom   
  void Randomize(const CuMatrix<float> &src, const CuStlVector<int32> &copy_from_idx, CuMatrix<float> *tgt);



  /*
   * Templated implementation to make it always compilable
   */
  template<typename Real>
  void Sigmoid(const CuMatrix<Real>& X, CuMatrix<Real>* Y) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  template<typename Real>
  void DiffSigmoid(const CuMatrix<Real>& Ein, const CuMatrix<Real>& Y, CuMatrix<Real>* Eout) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  template<typename Real>
  void Softmax(const CuMatrix<Real>& X, CuMatrix<Real>* Y) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }


  template<typename Real>
  void RegularizeL1(CuMatrix<Real> *wei, CuMatrix<Real> *grad, Real l1, Real lr) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }

 
  template<typename Real>
  void FindRowMaxId(const CuMatrix<Real> &mat, CuStlVector<int32> *id) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }
    
  template<typename Real>
  void DiffXent(const CuStlVector<int32> &tgt, CuMatrix<Real> *net_out_or_diff, CuVector<Real> *log_post_tgt) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  template<typename Real>
  void Randomize(const CuMatrix<Real> &src, const CuStlVector<int32> &copy_from_idx, CuMatrix<Real> *tgt) { 
    KALDI_ERR << __func__ << " Not implemented";
  }

} // namespace cu

} // namespace kaldi

#endif
