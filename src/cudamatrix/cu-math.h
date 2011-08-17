#ifndef KALDI_CUDAMATRIX_CUMATH_H_
#define KALDI_CUDAMATRIX_CUMATH_H_

#include "cudamatrix/cu-matrix.h"

#include "util/timer.h"
#include "cudamatrix/cu-device.h"

namespace kaldi {
  
  
/**
 * Hide the CUDA kernel ANSI-C wrappers to subnamespace cu::
 */
namespace cu {
  
  /*
   * Float version of functions
   */
  /// Y = Sigmoid(X)
  void Sigmoid(const CuMatrix<float>& X, CuMatrix<float>* Y);

  /// Eout = E(1-E) * Y
  void DiffSigmoid(const CuMatrix<float>& Ein, const CuMatrix<float>& Y, CuMatrix<float>* Eout);
    
  /// Y = Softmax(X)
  void Softmax(const CuMatrix<float>& X, CuMatrix<float>* Y);

  /// check match in the classification for Xentropy
  void CheckClass(const CuMatrix<float>& out, const CuMatrix<float> &des, CuVector<float>* match);

  /// apply the L1 regularization, sets zero to wei and grad elements on zero-crossing
  void RegularizeL1(CuMatrix<float>* wei, CuMatrix<float>* grad, float l1, float lr);






  /*
   * Templated implementation to make it always compilable
   */
  template<typename _ElemT>
  void Sigmoid(const CuMatrix<_ElemT>& X, CuMatrix<_ElemT>* Y) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  template<typename _ElemT>
  void DiffSigmoid(const CuMatrix<_ElemT>& Ein, const CuMatrix<_ElemT>& Y, CuMatrix<_ElemT>* Eout) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  template<typename _ElemT>
  void Softmax(const CuMatrix<_ElemT>& X, CuMatrix<_ElemT>* Y) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  template<typename _ElemT>
  void CheckClass(const CuMatrix<_ElemT>& out, const CuMatrix<_ElemT> &des, CuVector<float>& match) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  template<typename _ElemT>
  void RegularizeL1(CuMatrix<_ElemT>* wei, CuMatrix<_ElemT>* grad, _ElemT l1, _ElemT lr) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }
 
} //namespace cu

} //namespace kaldi

#endif
