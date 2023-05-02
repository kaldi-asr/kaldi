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
#include "base/kaldi-utils.h"

namespace {

constexpr const char* kMagicPrefix = "\x93NUMPY";
constexpr int kMagicLen = 6;

#if defined(__BIG_ENDIAN__)
constexpr bool kIsLittleEndian = false;
#else
constexpr bool kIsLittleEndian = true;
#endif

template <bool isLittleEndian>
struct EndianString {
  static constexpr const char* value = isLittleEndian ? "<" : ">";
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

template <typename>
struct SwapReal;

template <>
struct SwapReal<float> {
  void operator()(float& f) const { KALDI_SWAP4(f); }
};

template <>
struct SwapReal<double> {
  void operator()(double& d) const { KALDI_SWAP8(d); }
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

  uint8_t major = 0xff;
  uint8_t minor = 0xff;
  in.read(reinterpret_cast<char*>(&major), 1);
  in.read(reinterpret_cast<char*>(&minor), 1);
  if (!in) {
    KALDI_ERR << "Failed to read major and minor version";
  }

  KALDI_ASSERT(minor == 0);

  uint32_t header_len = 0;
  uint32_t expected_len = 0;
  switch (major) {
    case 1:
      header_len = ReadHeaderLen10(in);
      // 4 = 2 bytes version + 2 bytes length
      expected_len = kMagicLen + 4 + header_len;
      break;
    case 2:  // fall-through
    case 3:
      header_len = ReadHeaderLen20And30(in);
      // 6 = 2 bytes version + 4 bytes length
      expected_len = kMagicLen + 6 + header_len;
      break;
    default:
      KALDI_ERR << "Unsupported major version: " << major << "\n"
                << "Supported major versions are: 1 and 2";
      break;
  }

  // expected_len should be multiple of 64, i.e., 64 bytes aligned
  if (expected_len & (64 - 1)) {
    KALDI_ERR << "Expected length " << expected_len
              << " is not a multiple of 64.";
  }

  std::string header(header_len, 0);
  in.read(&header[0], header_len);
  bool is_little_endian = ParseHeader(header);

  num_elements_ =
      std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());

  if (data_) {
    delete[] data_;
  }

  data_ = new Real[num_elements_];
  in.read(reinterpret_cast<char*>(data_), sizeof(Real) * num_elements_);

  if (is_little_endian != kIsLittleEndian) {
    auto swap_real = SwapReal<Real>();
    for (auto& d : *this) {
      swap_real(d);
    }
  }
}

template <typename Real>
void NumpyArray<Real>::Write(std::ostream& out, bool binary) const {
  if (!out.good()) {
    KALDI_ERR << "Failed to write vector to stream: stream not good";
  }

  KALDI_ASSERT(binary == true);
  out.write(kMagicPrefix, kMagicLen);

  // we always save to version 1.0 .
  out.put(static_cast<uint8_t>(1));
  out.put(static_cast<uint8_t>(0));

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
    // (fangjun): we have to add a trailing comma here.
    // For example, (1) is not a tuple in python, but (1,) is.
    os << ",";
  }
  os << "), }";

  uint16_t header_len = os.str().size();
  uint16_t expected_header_len =
      (header_len + kMagicLen + 4 + 63) & (~(64 - 1));

  expected_header_len -= kMagicLen + 4;
  // -1 for the final `\n`
  int padding_len = expected_header_len - header_len - 1;
  for (int i = 0; i < padding_len; ++i) {
    os << ' ';
  }
  os << '\n';

  if (!kIsLittleEndian) {
    // since we always save to version 1.0, header len is of type uint16_t
    KALDI_SWAP2(expected_header_len);
  }

  out.write(reinterpret_cast<char*>(&expected_header_len), 2);
  out.write(const_cast<char*>(os.str().c_str()), os.str().size());
  out.write(reinterpret_cast<char*>(data_), sizeof(Real) * num_elements_);

  KALDI_ASSERT(out);  // out should be in a good state
}

template <typename Real>
NumpyArray<Real>::NumpyArray(const MatrixBase<Real>& m) {
  num_elements_ = m.NumRows() * m.NumCols();
  KALDI_ASSERT(num_elements_ > 0);
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
  KALDI_ASSERT(num_elements_ > 0);
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
bool NumpyArray<Real>::ParseHeader(const std::string& header) {
  KALDI_ASSERT(header[0] == '{');
  KALDI_ASSERT(header.back() == '\n');
  auto pos = header.rfind('}');
  KALDI_ASSERT(pos != header.npos);
  for (auto p = pos + 1; p + 1 < header.size(); ++p) {
    KALDI_ASSERT(header[p] == ' ');
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
  KALDI_ASSERT(descr.size() == 3);

  char endian_str = descr[0];
  KALDI_ASSERT(endian_str == '<' || endian_str == '>');

  bool is_little_endian = (endian_str == '<');

  auto data_type_str = descr.substr(1);

  std::string expected_data_type_str = DataTypeString<Real>::value;
  if (data_type_str != expected_data_type_str) {
    KALDI_ERR << "Expected data type str: " << expected_data_type_str << "\n"
              << "Actual data type str: " << data_type_str;
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

  return is_little_endian;
}

template <typename Real>
uint32_t NumpyArray<Real>::ReadHeaderLen10(std::istream& in) {
  // refer to
  // https://github.com/numpy/numpy/blob/master/numpy/lib/format.py#L102
  //
  // The next 2 bytes form a little-endian unsigned short int: the length of
  // the header data HEADER_LEN.

  uint16_t header_len = 0;
  if (!in.read(reinterpret_cast<char*>(&header_len), 2)) {
    KALDI_ERR << "Failed to read header len";
  }

  if (!kIsLittleEndian) {
    KALDI_SWAP2(header_len);
  }
  return header_len;
}

template <typename Real>
uint32_t NumpyArray<Real>::ReadHeaderLen20And30(std::istream& in) {
  uint32_t header_len = 0;
  // The next 4 bytes form a little-endian unsigned int: the length of
  // the header data HEADER_LEN.
  if (!in.read(reinterpret_cast<char*>(&header_len), 4)) {
    KALDI_ERR << "Failed to read header len";
  }

  if (!kIsLittleEndian) {
    KALDI_SWAP4(header_len);
  }
  return header_len;
}

template class NumpyArray<float>;
template class NumpyArray<double>;

}  // namespace kaldi
