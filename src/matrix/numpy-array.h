// matrix/numpy-array.h

// Copyright 2020   Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)

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

#ifndef KALDI_MATRIX_NUMPY_ARRAY_H_
#define KALDI_MATRIX_NUMPY_ARRAY_H_ 1

#include <iostream>
#include <vector>

#include "matrix/kaldi-matrix.h"
#include "matrix/kaldi-vector.h"

namespace kaldi {

/// \addtogroup matrix_group
/// @{

/** NumpyArray for reading *.npy files.
 *
 * This class implements the format described at
 * https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
 */
template <typename Real>
class NumpyArray {
 public:
  NumpyArray() = default;
  NumpyArray(const NumpyArray&) = delete;
  NumpyArray& operator=(const NumpyArray&) = delete;

  ~NumpyArray() { delete[] data_; }

  void Read(std::istream& in, bool binary);

  void Write(std::ostream& out, bool binary) const;

  int NumElements() const { return num_elements_; }
  const std::vector<int>& Shape() const { return shape_; }

  const Real* Data() const { return data_; }
  Real* Data() { return data_; }

  Real* begin() { return data_; }
  Real* end() { return data_ + num_elements_; }

  const Real* begin() const { return data_; }
  const Real* end() const { return data_ + num_elements_; }

  NumpyArray(const MatrixBase<Real>& m);
  NumpyArray(const VectorBase<Real>& v);
  operator SubVector<Real>();
  operator SubMatrix<Real>();

  Real operator[](int i) const { return data_[i]; }
  Real& operator[](int i) { return data_[i]; }

 private:
  // for version 1.0
  static uint32_t ReadHeaderLen10(std::istream& in);

  // for version 2.0 and 3.0
  static uint32_t ReadHeaderLen20And30(std::istream& in);

  // return true if the data is saved in little endian
  // return false if the data is saved in big endian
  bool ParseHeader(const std::string& header);

 private:
  std::vector<int> shape_;
  Real* data_ = nullptr;
  uint32_t num_elements_ = 0;
};

/// @} end of \addtogroup matrix_group

}  // namespace kaldi

#endif  // KALDI_MATRIX_NUMPY_ARRAY_H_
