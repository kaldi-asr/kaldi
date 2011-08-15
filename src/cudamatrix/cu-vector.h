#ifndef KALDI_CUDAMATRIX_CUVECTOR_H_
#define KALDI_CUDAMATRIX_CUVECTOR_H_

#include "matrix/kaldi-vector.h"

namespace kaldi {

template<typename _ElemT> class CuMatrix;

/**
 * Matrix for CUDA computing
 */
template<typename _ElemT>
class CuVector {
 typedef CuVector<_ElemT> ThisType;

 public:

  /// Default Constructor
  CuVector<_ElemT>()
   : dim_(0), data_(NULL) { 
  }
  /// Constructor with memory initialisation
  CuVector<_ElemT>(size_t dim)
   : dim_(0), data_(NULL) { 
    Resize(dim); 
  }

  /// Destructor
  ~CuVector() {
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
  ThisType&        CopyFrom(const CuVector<_ElemT>& src);
  ThisType&        CopyFrom(const Vector<_ElemT>& src);
  void             CopyTo(Vector<_ElemT>& dst) const;

  ThisType&        CopyFrom(const std::vector<_ElemT>& src);
  void             CopyTo(std::vector<_ElemT>& dst) const;

  void             Read(std::istream& is, bool binary);
  void             Write(std::ostream& os, bool binary) const;

  
  // Math operations
  //
  void SetZero();

  void Set(_ElemT value) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }

  void AddVec(_ElemT alpha, const CuVector<_ElemT>& vec, _ElemT beta=1.0) {
    KALDI_ERR << "__func__ Not implemented"; 
  }

  void AddColSum(_ElemT alpha, const CuMatrix<_ElemT>& mat, _ElemT beta=1.0) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }


  /// Accessor to non-GPU vector
  const VectorBase<_ElemT>& Vec() const {
    return vec_;
  }
  VectorBase<_ElemT>& Vec() {
    return vec_;
  }



private:
  size_t dim_;
  _ElemT* data_;

  Vector<_ElemT> vec_; ///< non-GPU backup vector
};


/// I/O
template<typename _ElemT>
std::ostream& operator << (std::ostream& out, const CuVector<_ElemT>& vec);
 
  
} //namespace


#include "cu-vector.tcc"

#endif
