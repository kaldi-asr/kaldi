// matrix/compressed-matrix.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)
//                 Frantisek Skala, Wei Shi

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_MATRIX_COMPRESSED_MATRIX_H_
#define KALDI_MATRIX_COMPRESSED_MATRIX_H_ 1

#include "kaldi-matrix.h"

namespace kaldi {

/// \addtogroup matrix_group
/// @{

/// This class does lossy compression of a matrix.  It only
/// supports copying to-from a KaldiMatrix.  For large matrices,
/// each element is compressed into about one byte, but there
/// is a little overhead on top of that (globally, and also per
/// column).

/// The basic idea is for each column (in the normal configuration)
/// we work out the values at the 0th, 25th, 50th and 100th percentiles
/// and store them as 16-bit integers; we then encode each value in
/// the column as a single byte, in 3 separate ranges with different
/// linear encodings (0-25th, 25-50th, 50th-100th).
/// If the matrix has 8 rows or fewer, we simply store all values as
/// uint16.

class CompressedMatrix {
 public:
  CompressedMatrix(): data_(NULL) { }

  ~CompressedMatrix() { Destroy(); }
  
  template<typename Real>
  CompressedMatrix(const MatrixBase<Real> &mat): data_(NULL) { CopyFromMat(mat); }

  /// Initializer that can be used to select part of an existing
  /// CompressedMatrix without un-compressing and re-compressing (note: unlike
  /// similar initializers for class Matrix, it doesn't point to the same memory
  /// location).
  CompressedMatrix(const CompressedMatrix &mat,
                   const MatrixIndexT row_offset,
                   const MatrixIndexT num_rows,
                   const MatrixIndexT col_offset,
                   const MatrixIndexT num_cols);

  void *Data() const { return this->data_; }

  /// This will resize *this and copy the contents of mat to *this.
  template<typename Real>
  void CopyFromMat(const MatrixBase<Real> &mat);

  CompressedMatrix(const CompressedMatrix &mat);

  CompressedMatrix &operator = (const CompressedMatrix &mat); // assignment operator.

  template<typename Real>
  CompressedMatrix &operator = (const MatrixBase<Real> &mat); // assignment operator.
  
  /// Copies contents to matrix.  Note: mat must have the correct size,
  /// CopyToMat no longer attempts to resize it.
  template<typename Real>
  void CopyToMat(MatrixBase<Real> *mat) const;

  void Write(std::ostream &os, bool binary) const;
  
  void Read(std::istream &is, bool binary);

  /// Returns number of rows (or zero for emtpy matrix).
  inline MatrixIndexT NumRows() const { return (data_ == NULL) ? 0 :
      (*reinterpret_cast<GlobalHeader*>(data_)).num_rows; }

  /// Returns number of columns (or zero for emtpy matrix).
  inline MatrixIndexT NumCols() const { return (data_ == NULL) ? 0 :
      (*reinterpret_cast<GlobalHeader*>(data_)).num_cols; }

  /// Copies row #row of the matrix into vector v.
  /// Note: v must have same size as #cols.
  template<typename Real>
  void CopyRowToVec(MatrixIndexT row, VectorBase<Real> *v) const;

  /// Copies column #col of the matrix into vector v.
  /// Note: v must have same size as #rows.
  template<typename Real>
  void CopyColToVec(MatrixIndexT col, VectorBase<Real> *v) const;

  /// Copies submatrix of compressed matrix into matrix dest.
  /// Submatrix starts at row row_offset and column column_offset and its size
  /// is defined by size of provided matrix dest
  template<typename Real>
  void CopyToMat(int32 row_offset,
                 int32 column_offset,
                 MatrixBase<Real> *dest) const;

  void Swap(CompressedMatrix *other) { std::swap(data_, other->data_); }
  
  friend class Matrix<float>;
  friend class Matrix<double>;
 private:

  // allocates data using new [], ensures byte alignment
  // sufficient for float.
  static void *AllocateData(int32 num_bytes);

  // the "format" will be 1 for the original format where each column has a
  // PerColHeader, and 2 for the format now used for matrices with 8 or fewer
  // rows, where everything is represented as 16-bit integers.
  struct GlobalHeader {
    int32 format;
    float min_value;
    float range;
    int32 num_rows;
    int32 num_cols;
  };

  static MatrixIndexT DataSize(const GlobalHeader &header);

  struct PerColHeader {
    uint16 percentile_0;
    uint16 percentile_25;
    uint16 percentile_75;
    uint16 percentile_100;
  };

  template<typename Real>
  static void CompressColumn(const GlobalHeader &global_header,
                             const Real *data, MatrixIndexT stride,
                             int32 num_rows, PerColHeader *header,
                             unsigned char *byte_data);
  template<typename Real>
  static void ComputeColHeader(const GlobalHeader &global_header,
                               const Real *data, MatrixIndexT stride,
                               int32 num_rows, PerColHeader *header);

  static inline uint16 FloatToUint16(const GlobalHeader &global_header,
                                     float value);

  static inline float Uint16ToFloat(const GlobalHeader &global_header,
                                    uint16 value);
  static inline unsigned char FloatToChar(float p0, float p25,
                                          float p75, float p100,
                                          float value);
  static inline float CharToFloat(float p0, float p25,
                                  float p75, float p100,
                                  unsigned char value);
  
  void Destroy();
  
  void *data_; // first GlobalHeader, then PerColHeader (repeated), then
  // the byte data for each column (repeated).  Note: don't intersperse
  // the byte data with the PerColHeaders, because of alignment issues.

};

enum GeneralMatrixType {
  int32 kFullMatrix,
  int32 kCompressedMatrix,
  int32 kSparseMatrix
};

/// This class is a wrapper that enables you to store a matrix
/// in one of three forms: either as a Matrix<BaseFloat>, or a CompressedMatrix,
/// or a SparseMatrix<BaseFloat>.  It handles the I/O for you, i.e. you read
/// and write a single object type.  It is useful for neural-net training targets
/// which might be sparse or not, and might be compressed or not.
class GeneralMatrix {
 public:
  GeneralMatrixType Type();

  void Compress();  // Must only be called if Type() == kFullMatrix.
                    // Compresses and changes Type() to kCompressedMatrix.

  void Uncompress();  // Must only be called if Type() == kCompressedMatrix.
                      // Uncompresses and changes Type() to kFullMatrix.
  
  void Write(std::ostream &os, bool binary) const;

  /// Note: if you write a compressed matrix in text form, it will be read as
  /// uncompressed.
  void Read(std::istream &is, bool binary);

  /// Outputs the contents as a SparseMatrix.  This will only work if
  /// Type() returns kSparseMatrix.
  void GetSparseMatrix(SparseMatrix<BaseFloat> *smat);

  /// Outputs the contents as a matrix.  This will work regardless of
  /// Type().
  void GetMatrix(Matrix<BaseFloat> *mat) const;

  /// Assignment from regular matrix.
  GeneralMatrix &operator = (const MatrixBase<BaseFloat> &mat);

  /// Assignment from compressed matrix.
  GeneralMatrix &operator = (CompressedMatrix &mat) const;

  /// Assignment from sparse matrix
  template <typename Real>
  GeneralMatrix &operator = (SparseMatrix<BaseFloat> &smat) const;

  int32 NumRows() const;

  int32 NumCols() const;

  GeneralMatrix(const MatrixBase<BaseFloat> &mat) { *this = mat; }

  GeneralMatrix(const CompressedMatrix &cmat) { *this = cmat; }

  GeneralMatrix(const SparseMatrix<BaseFloat> &smat) { *this = smat; }
  
  GeneralMatrix() { }
  // Copy constructor
  GeneralMatrix(const GeneralMatrix &other);
  // Assignment operator.
  GeneralMatrix &operator =(const GeneralMatrix &other);
  // Sets to the empty matrix.
  void Clear(); 
 private:
  // We don't explicitly store the type of the matrix.  Rather, we make
  // sure that only one of the matrices is ever nonempty, and the Type()
  // returns that one, or kFullMatrix if all are empty.
  Matrix<BaseFloat> mat_;
  CompressedMatrix cmat_;
  SparseMatrix<BaseFloat> smat_;
};


/// Appends all the matrix rows of a list of GeneralMatrixes, to get a single
/// GeneralMatrix.  Preserves sparsity if all inputs were sparse.
void AppendMatrixRows(const std::vector<const GeneralMatrix *> &src,
                      GeneralMatrix *mat);

/// @} end of \addtogroup matrix_group


}  // namespace kaldi


#endif  // KALDI_MATRIX_COMPRESSED_MATRIX_H_
