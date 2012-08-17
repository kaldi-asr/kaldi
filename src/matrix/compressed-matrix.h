// matrix/compressed-matrix.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

class CompressedMatrix {
 public:
  CompressedMatrix(): data_(NULL) { }

  template<class Real>
  CompressedMatrix(const Matrix<Real> &mat): data_(NULL) { CopyFromMat(mat); }
  
  template<class Real>
  void CopyFromMat(const Matrix<Real> &mat);

  /// Caution: CopyToMat will resize the output matrix if necessary.
  template<class Real>
  void CopyToMat(Matrix<Real> *mat) const;

  
  void Write(std::ostream &os, bool binary) const;
  
  void Read(std::istream &is, bool binary);

  /// Returns number of rows (or zero for emtpy matrix).
  inline int32 NumRows() const { return (data_ == NULL) ? 0 : 
      (*reinterpret_cast<GlobalHeader*>(data_)).num_rows; }

  /// Returns number of columns (or zero for emtpy matrix).
  inline int32 NumCols() const { return (data_ == NULL) ? 0 : 
      (*reinterpret_cast<GlobalHeader*>(data_)).num_cols; }
  
  friend class Matrix<float>;
  friend class Matrix<double>;
  private:
  
  // allocates data using new [], ensures byte alignment
  // sufficient for float.
  static unsigned char *AllocateData(int32 num_bytes);
  
  struct GlobalHeader {
    float min_value;
    float range;
    int32 num_rows;
    int32 num_cols;
  };
  
  static MatrixIndexT DataSize(const GlobalHeader &header) {
    // Returns size in bytes of the data.
    return sizeof(GlobalHeader) +
        header.num_cols * (sizeof(PerColHeader) + header.num_rows);
  }  

  struct PerColHeader {
    uint16 percentile_0;
    uint16 percentile_25;
    uint16 percentile_75;
    uint16 percentile_100;
  };

  // The following function is called in CopyToMatrix.
  template<class Real>
  static void CompressColumn(const GlobalHeader &global_header,
                             const Real *data, MatrixIndexT stride,
                             int32 num_rows, PerColHeader *header,
                             unsigned char *byte_data);
  template<class Real>
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

  unsigned char *data_; // first GlobalHeader, then PerColHeader (repeated), then
  // the byte data for each column (repeated).  Note: don't intersperse
  // the byte data with the PerColHeaders, because of alignment issues.
};


/// @} end of \addtogroup matrix_group


}  // namespace kaldi


#endif  // KALDI_MATRIX_COMPRESSED_MATRIX_H_
