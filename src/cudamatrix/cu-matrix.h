#ifndef KALDI_CUDAMATRIX_CUMATRIX_H_
#define KALDI_CUDAMATRIX_CUMATRIX_H_

#include <sstream>

#include "cudamatrix/cu-matrixdim.h"

#include "matrix/matrix-common.h"
#include "matrix/kaldi-matrix.h"


namespace kaldi {

template<typename Real> class CuVector;

/**
 * Matrix for CUDA computing,
 *
 * It has "polymorphic" behavior. When CUDA is not compiled in 
 * or is not Enabled() the computation is back-off'ed to the CPU.
 */
template<typename Real>
class CuMatrix {
 typedef CuMatrix<Real> ThisType;
  
 public:

  /// Default Constructor
  CuMatrix<Real>()
   : num_rows_(0), num_cols_(0), stride_(0), data_(NULL) { 
  }
  /// Constructor with memory initialisation
  CuMatrix<Real>(MatrixIndexT rows, MatrixIndexT cols)
   : num_rows_(0), num_cols_(0), stride_(0), data_(NULL) { 
    Resize(rows, cols); 
  }

  /// Destructor
  ~CuMatrix() {
    Destroy(); 
  }

  /// Dimensions
  MatrixIndexT NumRows() const { 
    return num_rows_; 
  }

  MatrixIndexT NumCols() const { 
    return num_cols_; 
  }

  MatrixIndexT Stride() const { 
    return stride_; 
  }

  ::MatrixDim Dim() const { 
    ::MatrixDim d = { num_rows_, num_cols_, stride_ }; 
    return d; 
  }

  /// Get raw pointer
  const Real* Data() const;
  Real* Data();
  
  /// Get raw row pointer
  const Real* RowData(MatrixIndexT r) const;
  Real* RowData(MatrixIndexT r);

  /// Get size of matrix in bytes
  MatrixIndexT SizeInBytes() const { 
    return num_rows_*stride_*sizeof(Real); 
  }
  
  /// Get size of matrix row in bytes
  MatrixIndexT RowSizeInBytes() const {
    return num_cols_*sizeof(Real); 
  }
  
  /// Get size of matrix stride in bytes
  MatrixIndexT StrideSizeInBytes() const {
    return stride_*sizeof(Real); 
  }

  /// Allocate the memory
  ThisType& Resize(MatrixIndexT rows, MatrixIndexT cols);

  /// Deallocate the memory
  void Destroy();

  /// Copy functions (reallocates when needed)
  ThisType&        CopyFromMat(const CuMatrix<Real> &src);
  ThisType&        CopyFromMat(const Matrix<Real> &src);
  void             CopyToMat(Matrix<Real> *dst) const;

  /// Copy row interval from matrix
  /// @param r      [in] number of rows to copy.
  /// @param src    [in] source matrix.
  /// @param src_ro [in] source matrix row offset.
  /// @param dst_ro [in] destination matrix row offset.
  void             CopyRowsFromMat(int32 r, const CuMatrix<Real> &src, int32 src_ro, int32 dst_ro);

  /// I/O functions
  void             Read(std::istream &is, bool binary);
  void             Write(std::ostream &os, bool binary) const;


  // Math operations, some calling kernels
  //
  void SetZero();
  void Set(Real value) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }

  void ApplyLog() { 
    KALDI_ERR << "__func__ Not implemented"; 
  }

  /// Multiply two matrices elementhwise: C = A .* C
  void MulElements(const CuMatrix<Real>& A) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }

  /// scale i'th column by scale[i]
  void MulColsVec(const CuVector<Real> &scale) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }

  /// scale i'th row by scale[i]
  void MulRowsVec(const CuVector<Real> &scale) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }

  /// divide i'th row by scale[i]
  void DivRowsVec(const CuVector<Real> &div) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }
  
  /// B = aplha * A + beta * B
  void AddMat(Real alpha, const CuMatrix<Real>& A, Real beta=1.0) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }

  /// B = aplha * row + beta * B
  void AddScaledRow(Real alpha, const CuVector<Real> &row, Real beta=1.0) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }

  /// C = alpha * A(^T)*B(^T) + beta * C
  void AddMatMat(Real alpha, const CuMatrix<Real>& A, MatrixTransposeType transA,
                 const CuMatrix<Real>& B, MatrixTransposeType transB, Real beta) { 
    KALDI_ERR << "__func__ Not implemented"; 
  }

  /// Accessor to the non-CUDA matrix
  const MatrixBase<Real>& Mat() const {
    return mat_;
  }
  MatrixBase<Real>& Mat() {
    return mat_;
  }

 private:
  MatrixIndexT num_rows_;
  MatrixIndexT num_cols_;
  MatrixIndexT stride_;

  Real *data_;       ///< GPU data pointer
  
  Matrix<Real> mat_; ///< non-GPU matrix as back-off


}; // class CuMatrix


/// I/O
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuMatrix<Real> &mat);

  
} // namespace


#include "cu-matrix-inl.h"

#endif
