#ifndef KALDI_CUDAMATRIX_CUVECTOR_H_
#define KALDI_CUDAMATRIX_CUVECTOR_H_

#include "matrix/kaldi-vector.h"

namespace kaldi {

template<typename Real> class CuMatrix;

/**
 * Vector for CUDA computing
 */
template<typename Real>
class CuVector {
 typedef CuVector<Real> ThisType;

 public:

  /// Default Constructor
  CuVector<Real>()
   : dim_(0), data_(NULL) { 
  }
  /// Constructor with memory initialisation
  CuVector<Real>(size_t dim)
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
  const Real* Data() const;
  Real* Data();
 
  /// Allocate the memory
  ThisType& Resize(size_t dim);

  /// Deallocate the memory
  void Destroy();

  /// Copy functions (reallocates when needed)
  ThisType&        CopyFromVec(const CuVector<Real> &src);
  ThisType&        CopyFromVec(const Vector<Real> &src);
  void             CopyToVec(Vector<Real> *dst) const;

  void             Read(std::istream &is, bool binary);
  void             Write(std::ostream &is, bool binary) const;
  
  // Math operations
  //
  void SetZero();

  void Set(Real value) { 
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  void AddVec(Real alpha, const CuVector<Real> &vec, Real beta=1.0) {
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  void AddColSum(Real alpha, const CuMatrix<Real> &mat, Real beta=1.0) { 
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  void AddRowSum(Real alpha, const CuMatrix<Real> &mat, Real beta=1.0) { 
    KALDI_ERR << __func__ << " Not implemented"; 
  }

  void InvertElements() {
    KALDI_ERR << __func__ << " Not implemented"; 
  }



  /// Accessor to non-GPU vector
  const VectorBase<Real>& Vec() const {
    return vec_;
  }
  VectorBase<Real>& Vec() {
    return vec_;
  }



private:
  size_t dim_;
 
  Real *data_; ///< GPU data pointer

  Vector<Real> vec_; ///< non-GPU vector as back-off
};


/// I/O
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuVector<Real> &vec);
 
  
} // namespace


#include "cu-vector-inl.h"

#endif
