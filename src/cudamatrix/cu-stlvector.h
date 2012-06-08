#ifndef KALDI_CUDAMATRIX_CUSTLVECTOR_H_
#define KALDI_CUDAMATRIX_CUSTLVECTOR_H_

#include "matrix/kaldi-vector.h"

namespace kaldi {

template<typename IntType> class CuMatrix;

/**
 * Vector for CUDA computing
 */
template<typename IntType>
class CuStlVector {
 typedef CuStlVector<IntType> ThisType;

 public:

  /// Default Constructor
  CuStlVector<IntType>()
   : dim_(0), data_(NULL) { 
  }
  /// Constructor with memory initialisation
  CuStlVector<IntType>(size_t dim)
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
  const IntType* Data() const;
  IntType* Data();
 
  /// Allocate the memory
  ThisType& Resize(size_t dim);

  /// Deallocate the memory
  void Destroy();

  /// Copy functions (reallocates when needed)
  ThisType&        CopyFromVec(const std::vector<IntType> &src);
  void             CopyToVec(std::vector<IntType> *dst) const;
  
  // Math operations
  //
  void SetZero();

  void Set(IntType value) { 
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  /// Accessor to non-GPU vector
  const std::vector<IntType>& Vec() const {
    return vec_;
  }
  std::vector<IntType>& Vec() {
    return vec_;
  }



private:
  size_t dim_;
 
  IntType *data_; ///< GPU data pointer

  std::vector<IntType> vec_; ///< non-GPU vector as back-off
};


/// I/O
template<typename IntType>
std::ostream &operator << (std::ostream &out, const CuStlVector<IntType> &vec);
 
  
} // namespace


#include "cu-stlvector-inl.h"

#endif
