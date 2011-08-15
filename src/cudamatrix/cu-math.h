#ifndef KALDI_CUDAMATRIX_CUMATH_H_
#define KALDI_CUDAMATRIX_CUMATH_H_

#include "cudamatrix/cu-matrix.h"

#include "util/timer.h"
#include "cudamatrix/cu-device.h"

namespace kaldi {
  
  
/**
 * Grouping class for various CUDA kernel wrappers...
 */
template<typename _ElemT>
class CuMath {
 public:

  /// Y = Sigmoid(X)
  static void Sigmoid(CuMatrix<_ElemT>& Y, const CuMatrix<_ElemT>& X) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }

  /// Eout = E(1-E) * Y
  static void DiffSigmoid(CuMatrix<_ElemT>& Eout, const CuMatrix<_ElemT>& Ein, const CuMatrix<_ElemT>& Y) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }

  /// Y = Softmax(X)
  static void Softmax(CuMatrix<_ElemT>& Y, const CuMatrix<_ElemT>& X) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }

  /// check match in the classification for Xentropy
  static void CheckClass(const CuMatrix<_ElemT>& out, const CuMatrix<_ElemT> &des, CuVector<float>& match) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }
  
}; //class CuMath::


//////////////////////////////////////////////////////////////////////////////
//// CuMath<> Template specializations (float)
////
template<>
void CuMath<float>::Sigmoid(CuMatrix<float>& Y, const CuMatrix<float>& X);

template<>
void CuMath<float>::DiffSigmoid(CuMatrix<float>& Eout, const CuMatrix<float>& Ein, const CuMatrix<float>& Y);
  
template<>
void CuMath<float>::Softmax(CuMatrix<float>& Y, const CuMatrix<float>& X);

template<>
void CuMath<float>::CheckClass(const CuMatrix<float>& out, const CuMatrix<float> &des, CuVector<float>& match);

} //namespace

#endif
