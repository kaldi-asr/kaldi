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

  /// Find the id of the maximal element for each row
  void FindRowMaxId(const CuMatrix<float>& mat, CuStlVector<int32>* id);

  /// Differentiate cross-entropy+softmax coupling (subtract post - tgt)
  /// extract per-frame cross-entropy to vector log_post_tgt_
  void DiffXent(const CuStlVector<int32>& tgt, CuMatrix<BaseFloat>* net_out_or_diff, CuVector<BaseFloat>* log_post_tgt_);

  /// Sum each row of the matrix
  void SumRowsVec(const CuMatrix<BaseFloat>& mat, CuVector<BaseFloat>* sum);

  /// ie. switch rows according to copyFrom   
  void Randomize(const CuMatrix<BaseFloat>& src, const CuStlVector<int32>& copy_from_idx, CuMatrix<BaseFloat>* tgt);

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

 
  template<typename _ElemT>
  void FindRowMaxId(const CuMatrix<float>& mat, CuStlVector<int32>* id) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }
    
  template<typename _ElemT>
  void DiffXent(const CuStlVector<int32>& tgt, CuMatrix<_ElemT>* net_out_or_diff, CuVector<_ElemT>* log_post_tgt_) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  template<typename _ElemT>
  void SumRowsVec(const CuMatrix<_ElemT>& mat, CuVector<_ElemT>* sum) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  template<typename _ElemT>
  void Randomize(const CuMatrix<_ElemT>& src, const CuStlVector<int32>& copy_from_idx, CuMatrix<_ElemT>* tgt) { 
    KALDI_ERR << __func__ << " Not implemented";
  }

} // namespace cu

} // namespace kaldi

#endif
