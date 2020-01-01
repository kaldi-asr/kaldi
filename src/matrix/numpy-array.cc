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

#include "matrix/numpy-array.h"

#include <functional>  // for std::multiplies
#include <numeric>     // for std::accumulate

#include "base/kaldi-error.h"

namespace {

constexpr const char* kMagicPrefix = "\x93NUMPY";
constexpr int kMagicLen = 6;
// TODO(fangjun): We support only format 1.0 at present.
// Support for 2.0 and 3.0 can be added later when needed.
constexpr int8_t kMajorVersion = 1;
constexpr int8_t kMinorVersion = 0;

constexpr bool kIsLittleEndian = true;

// TODO(fangjun): Get `kIsLittleEndian` at runtime instead of at compile time.

template <bool is_little_endian>
struct EndianString {
  // for little endian
  static constexpr const char* value = "<";
};

template <>
struct EndianString<false> {
  // for big endian
  static constexpr const char* value = ">";
};

template <typename>
struct DataTypeString;

template <>
struct DataTypeString<float> {
  static constexpr const char* value = "f4";
};

template <>
struct DataTypeString<double> {
  static constexpr const char* value = "f8";
};

}  // namespace

namespace kaldi {

template <typename Real>
void NumpyArray<Real>::Read(std::istream& in, bool binary) {
  KALDI_ASSERT(binary == true);
  std::string magic_prefix(kMagicLen + 1, 0);
  if (!in.read(&magic_prefix[0], kMagicLen)) {
    KALDI_ERR << "Failed to read the magic prefix";
  }
  if (strncmp(magic_prefix.c_str(), kMagicPrefix, kMagicLen)) {
    KALDI_ERR << "Expected prefix: " << kMagicPrefix << "\n"
              << "Actual prefix: " << magic_prefix;
  }

  int8_t major = -1;
  int8_t minor = -1;
  in.read(reinterpret_cast<char*>(&major), 1);
  in.read(reinterpret_cast<char*>(&minor), 1);
  if (!in) {
    KALDI_ERR << "Failed to read major and minor version";
  }

  if (major != kMajorVersion || minor != kMinorVersion) {
    KALDI_ERR << "Exepcted version: " << (int)kMajorVersion << "."
              << (int)kMinorVersion << "\n"
              << "Actual version: " << (int)major << "." << (int)minor;
  }

  int16_t header_len = -1;
  if (!in.read(reinterpret_cast<char*>(&header_len), 2)) {
    KALDI_ERR << "Failed to read header len";
  }

  if (!kIsLittleEndian) {
    // TODO(fangjun): we should swap the bytes in header_len
  }

  int expected_len = kMagicLen + 4 + header_len;
  if (expected_len & 0x0f) {
    KALDI_ERR << "Expected length " << expected_len
              << " is not a multiple of 16.";
  }

  std::string header(header_len, 0);
  in.read(&header[0], header_len);
  ParseHeader(header);

  num_elements_ =
      std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());

  if (data_) {
    delete[] data_;
  }

  data_ = new Real[num_elements_];
  in.read(reinterpret_cast<char*>(data_), sizeof(Real) * num_elements_);
}

template <typename Real>
void NumpyArray<Real>::Write(std::ostream& out, bool binary) const {
  if (!out.good()) {
    KALDI_ERR << "Failed to write vector to stream: stream not good";
  }

  KALDI_ASSERT(binary == true);
  out.write(kMagicPrefix, kMagicLen);
  out.put(kMajorVersion);
  out.put(kMinorVersion);

  std::ostringstream os;
  os << "{";
  os << "'descr': '" << EndianString<kIsLittleEndian>::value
     << DataTypeString<Real>::value << "', ";
  os << "'fortran_order': False, ";
  os << "'shape': (";
  std::string s = "";
  for (auto d : shape_) {
    os << s << d;
    s = ", ";
  }
  if (shape_.size() == 1) {
    os << ",";
  }
  os << "), }";

  int16_t header_len = os.str().size();
  int16_t expected_header_len = (header_len + kMagicLen + 4 + 15) & (~0x0f);

  expected_header_len -= kMagicLen + 4;
  int padding_len = expected_header_len - header_len - 1;
  for (int i = 0; i < padding_len; ++i) {
    os << ' ';
  }
  os << '\n';

  // WARNING(fangjun): we assume it is little endian
  out.write(reinterpret_cast<char*>(&expected_header_len), 2);
  out.write(const_cast<char*>(os.str().c_str()), os.str().size());
  out.write(reinterpret_cast<char*>(data_), sizeof(Real) * num_elements_);
}

template <typename Real>
NumpyArray<Real>::NumpyArray(const MatrixBase<Real>& m) {
  num_elements_ = m.NumRows() * m.NumCols();
  shape_.resize(2);
  shape_[0] = m.NumRows();
  shape_[1] = m.NumCols();

  data_ = new Real[num_elements_];
  auto* dst = data_;
  for (int i = 0; i < m.NumRows(); ++i, dst += m.NumCols()) {
    memcpy(dst, m.RowData(i), sizeof(Real) * m.NumCols());
  }
}

template <typename Real>
NumpyArray<Real>::NumpyArray(const VectorBase<Real>& v) {
  num_elements_ = v.Dim();
  shape_.resize(1);
  shape_[0] = v.Dim();
  data_ = new Real[num_elements_];
  memcpy(data_, v.Data(), sizeof(Real) * v.Dim());
}

template <typename Real>
NumpyArray<Real>::operator SubVector<Real>() {
  SubVector<Real> sub(data_, num_elements_);
  return sub;
}

template <typename Real>
NumpyArray<Real>::operator SubMatrix<Real>() {
  KALDI_ASSERT(shape_.size() == 2);
  SubMatrix<Real> sub(data_, shape_[0], shape_[1], shape_[1]);
  return sub;
}

template <typename Real>
void NumpyArray<Real>::ParseHeader(const std::string& header) {
  KALDI_ASSERT(header[0] == '{');
  KALDI_ASSERT(header.back() == '\n');
  auto pos = header.rfind('}');
  KALDI_ASSERT(pos != header.npos);
  for (auto p = pos + 1; p + 1 < header.size(); ++p) {
    KALDI_ASSERT(header[p] == '\x20');
  }

  // for `descr`
  auto start_pos = header.find("'descr'");
  KALDI_ASSERT(start_pos != header.npos);
  start_pos += 7;

  start_pos = header.find("'", start_pos);
  KALDI_ASSERT(start_pos != header.npos);
  start_pos += 1;

  auto end_pos = header.find("'", start_pos);

  auto descr = header.substr(start_pos, end_pos - start_pos);
  // WARNING(fangjun): we support only little endian
  std::string expected_descr =
      std::string(EndianString<kIsLittleEndian>::value) +
      DataTypeString<Real>::value;
  if (descr != expected_descr) {
    KALDI_ERR << "Expected descr: " << expected_descr << "\n"
              << "Actual descr: " << descr;
  }

  // for `fortran_order`
  start_pos = header.find("'fortran_order': ");
  KALDI_ASSERT(start_pos != header.npos);
  start_pos += 17;

  end_pos = header.find(",", start_pos);
  KALDI_ASSERT(end_pos != header.npos);

  auto fortran_order = header.substr(start_pos, end_pos - start_pos);

  // WARNING(fangjun): we support only C order
  KALDI_ASSERT(fortran_order == "False");

  // for `shape`
  start_pos = header.find("'shape': ");
  KALDI_ASSERT(start_pos != header.npos);

  start_pos = header.find("(", start_pos);

  end_pos = header.find(")", start_pos);
  KALDI_ASSERT(end_pos != header.npos);
  auto shape = header.substr(start_pos, end_pos - start_pos + 1);

  char tmp;
  shape_.clear();
  std::stringstream ss(shape);
  while (ss >> tmp && tmp != ')') {
    int d;
    ss >> d;
    if (!ss) break;
    shape_.push_back(d);
  }
  if (shape_.size() != 1 && shape_.size() != 2) {
    KALDI_ERR << "Expected shape size: 1 or 2\n"
              << "Actual shape size: " << shape_.size();
  }
}

template class NumpyArray<float>;
template class NumpyArray<double>;

}  // namespace kaldi
