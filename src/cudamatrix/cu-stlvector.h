#ifndef KALDI_CUDAMATRIX_CUSTLVECTOR_H_
#define KALDI_CUDAMATRIX_CUSTLVECTOR_H_

#include "matrix/kaldi-vector.h"

namespace kaldi {

template<typename _ElemT> class CuMatrix;

/**
 * Vector for CUDA computing
 */
template<typename _ElemT>
class CuStlVector {
 typedef CuStlVector<_ElemT> ThisType;

 public:

  /// Default Constructor
  CuStlVector<_ElemT>()
   : dim_(0), data_(NULL) { 
  }
  /// Constructor with memory initialisation
  CuStlVector<_ElemT>(size_t dim)
   : dim_(0), data_(NULL) { 
    Resize(dim); 
  }

  /// Destructor
  ~CuStlVector() {
    Destroy(); 
  }

  /// Dimensions
  size_t Dim() const { 
    return dim_; 
  }

  /// Get raw pointer
  const _ElemT* Data() const;
  _ElemT* Data();
 
  /// Allocate the memory
  ThisType& Resize(size_t dim);

  /// Deallocate the memory
  void Destroy();

  /// Copy functions (reallocates when needed)
  ThisType&        CopyFromVec(const std::vector<_ElemT> &src);
  void             CopyToVec(std::vector<_ElemT> *dst) const;
  
  // Math operations
  //
  void SetZero();

  void Set(_ElemT value) { 
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  /// Accessor to non-GPU vector
  const std::vector<_ElemT>& Vec() const {
    return vec_;
  }
  std::vector<_ElemT>& Vec() {
    return vec_;
  }



private:
  size_t dim_;
 
  _ElemT *data_; ///< GPU data pointer

  std::vector<_ElemT> vec_; ///< non-GPU vector as back-off
};


/// I/O
template<typename _ElemT>
std::ostream &operator << (std::ostream &out, const CuStlVector<_ElemT> &vec);
 
  
} // namespace


#include "cu-stlvector-inl.h"

#endif
