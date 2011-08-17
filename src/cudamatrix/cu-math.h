#ifndef KALDI_CUDAMATRIX_CUMATH_H_
#define KALDI_CUDAMATRIX_CUMATH_H_

#include "cudamatrix/cu-matrix.h"

#include "util/timer.h"
#include "cudamatrix/cu-device.h"

namespace kaldi {
  
  
/**
 * Separate CUDA kernel wrappers to subnamespace...
 */
namespace cu {
  
  /*
   * Float version of functions
   */
  /// Y = Sigmoid(X)
  void Sigmoid(CuMatrix<float>& Y, const CuMatrix<float>& X);

  /// Eout = E(1-E) * Y
  void DiffSigmoid(CuMatrix<float>& Eout, const CuMatrix<float>& Ein, const CuMatrix<float>& Y);
    
  /// Y = Softmax(X)
  void Softmax(CuMatrix<float>& Y, const CuMatrix<float>& X);

  /// check match in the classification for Xentropy
  void CheckClass(const CuMatrix<float>& out, const CuMatrix<float> &des, CuVector<float>& match);


  /*
   * Templated implementation to make it always compilable
   */
  template<typename _ElemT>
  void Sigmoid(CuMatrix<_ElemT>& Y, const CuMatrix<_ElemT>& X) { 
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  template<typename _ElemT>
  void DiffSigmoid(CuMatrix<_ElemT>& Eout, const CuMatrix<_ElemT>& Ein, const CuMatrix<_ElemT>& Y) { 
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  template<typename _ElemT>
  void Softmax(CuMatrix<_ElemT>& Y, const CuMatrix<_ElemT>& X) { 
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  template<typename _ElemT>
  void CheckClass(const CuMatrix<_ElemT>& out, const CuMatrix<_ElemT> &des, CuVector<float>& match) { 
    KALDI_ERR << __func__ << " Not implemented"; 
  }

 
} //namespace Cu

} //namespace kaldi

#endif
