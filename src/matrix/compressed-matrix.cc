// matrix/compressed-matrix.cc

// Copyright 2012    Johns Hopkins University (author: Daniel Povey)

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

#include "matrix/compressed-matrix.h"
#include "matrix/sp-matrix.h"
#include "matrix/jama-svd.h"
#include "matrix/jama-eig.h"

namespace kaldi {

template<class Real>
void CompressedMatrix::CopyFromMat(
    const Matrix<Real> &mat) {
  if (data_ != NULL) {
    delete [] data_; // call delete [] because was allocated with new float[]
    data_ = NULL;
  }
  if (mat.NumRows() == 0) { return; } // Zero-size matrix stored as zero pointer.
  
  GlobalHeader global_header;
  KALDI_COMPILE_TIME_ASSERT(sizeof(global_header) == 16); // otherwise
  // something weird is happening and our code probably won't work or
  // won't be robust across platforms.

  // Below, the point of the "safety_margin" is that the minimum
  // and maximum values in the matrix shouldn't coincide with
  // the minimum and maximum ranges of the 16-bit range, because
  // this could cause certain problems in ComputeColHeader, where
  // we need to ensure that the percentile_0 through percentile_100
  // are in strictly increasing order. 
  float min_value = mat.Min(), max_value = mat.Max();
  float safety_margin = 0.001 * (fabs(min_value) + fabs(max_value));
  if (safety_margin == 0.0) safety_margin = 1.0;
  min_value -= safety_margin;
  max_value += safety_margin;
  
  global_header.min_value = min_value;
  global_header.range = max_value - min_value;
  // We can't compress the matrix if there are inf's or nan's.
  // The caller should check for this first.
  KALDI_ASSERT(KALDI_ISFINITE(global_header.min_value) &&
               KALDI_ISFINITE(global_header.range));
  
  // Avoid division by zero if the matrix is just a constant:
  // make sure max_value > min_value.
  if (global_header.range <= 0.0)
    global_header.range = 1.0e-05;
  global_header.num_rows = mat.NumRows();
  global_header.num_cols = mat.NumCols();
  
  int32 data_size = DataSize(global_header);
  
  data_ = AllocateData(data_size);
  
  *(reinterpret_cast<GlobalHeader*>(data_)) = global_header;

  PerColHeader *header_data =
      reinterpret_cast<PerColHeader*>(data_ + sizeof(GlobalHeader));
  unsigned char *byte_data =
      reinterpret_cast<unsigned char*>(header_data + global_header.num_cols);
  
  const Real *matrix_data = mat.Data();
  
  for (int32 col = 0; col < global_header.num_cols; col++) {
    CompressColumn(global_header,
                   matrix_data + col, mat.Stride(),
                   global_header.num_rows,
                   header_data, byte_data);
    header_data++;
    byte_data += global_header.num_rows;
  }
}

// Instantiate the template for float and double.
template
void CompressedMatrix::CopyFromMat(const Matrix<float> &mat);

template
void CompressedMatrix::CopyFromMat(const Matrix<double> &mat);

inline uint16 CompressedMatrix::FloatToUint16(
    const GlobalHeader &global_header,
    float value) {
  float f = (value - global_header.min_value) /
      global_header.range;
  if (f > 1.0) f = 1.0; // Note: this should not happen.
  if (f < 0.0) f = 0.0; // Note: this should not happen.
  return static_cast<int>(f * 65535 + 0.499); // + 0.499 is to
  // round to closest int; avoids bias.
}

inline float CompressedMatrix::Uint16ToFloat(
        const GlobalHeader &global_header,
        uint16 value) {
  // the constant 1.52590218966964e-05 is 1/65535.
  return global_header.min_value
      + global_header.range * 1.52590218966964e-05 * value;
}

      
template<class Real> // static
void CompressedMatrix::ComputeColHeader(
    const GlobalHeader &global_header,
    const Real *data, MatrixIndexT stride,
    int32 num_rows, CompressedMatrix::PerColHeader *header) {
  assert(num_rows > 0);
  std::vector<Real> sorted_data(num_rows);
  for (size_t i = 0, size = sorted_data.size(); i < size; i++)
    sorted_data[i] = data[i*stride];
  // Note: might be more efficient to use repeated calls to nth_element, rather
  // than sort().  If this proves to be using up a lot of CPU, we'll do that.
  std::sort(sorted_data.begin(), sorted_data.end());
  
  if (num_rows >= 5) {
    int quarter_num_rows = num_rows/4;
    header->percentile_0 = FloatToUint16(global_header, sorted_data[0]);
    header->percentile_25 = std::max<uint16>(
        FloatToUint16(global_header, sorted_data[quarter_num_rows]),
        header->percentile_0 + static_cast<uint16>(1));
    header->percentile_75 = std::max<uint16>(
        FloatToUint16(global_header, sorted_data[3*quarter_num_rows]),
        header->percentile_25 + static_cast<uint16>(1));
    header->percentile_100 = std::max<uint16>(
        FloatToUint16(global_header, sorted_data[num_rows-1]),
        header->percentile_75 + static_cast<uint16>(1));
  } else { // handle this pathological case.
    // Note: we know num_rows is at least 1.
    header->percentile_0 = FloatToUint16(global_header, sorted_data[0]);
    if (num_rows > 1)
      header->percentile_25 =
          std::max<uint16>(FloatToUint16(global_header, sorted_data[1]),
                           header->percentile_0 + 1);
    else
      header->percentile_25 = header->percentile_0 + 1;
    if (num_rows > 2)
      header->percentile_75 =
          std::max<uint16>(FloatToUint16(global_header, sorted_data[2]),
                           header->percentile_25 + 1);
    else
      header->percentile_75 = header->percentile_25 + 1;
    if (num_rows > 3)
      header->percentile_100 =
          std::max<uint16>(FloatToUint16(global_header, sorted_data[3]),
                           header->percentile_75 + 1);
    else
      header->percentile_100 = header->percentile_75 + 1;
  }    
}

// static
inline unsigned char CompressedMatrix::FloatToChar(
    float p0, float p25, float p75, float p100,
    float value) {
  int ans;
  if (value < p25) { // range [ p0, p25 ) covered by
    // characters 0 .. 64.  We round to the closest int.
    float f = (value - p0) / (p25 - p0);
    ans = static_cast<int>(f * 64 + 0.5);
    // Note: the checks on the next two lines
    // are necessary in pathological cases when all the elements in a row
    // are the same and the percentile_* values are separated by one.
    if (ans < 0) ans = 0;
    if (ans > 64) ans = 64;
  } else if (value < p75) { // range [ p25, p75 )covered
    // by characters 64 .. 192.  We round to the closest int.
    float f = (value - p25) / (p75 - p25);
    ans = 64 + static_cast<int>(f * 128 + 0.5);
    if (ans < 64) ans = 64;
    if (ans > 192) ans = 192;
  } else { // range [ p75, p100 ] covered by
    // characters 192 .. 255.  Note: this last range
    // has fewer characters than the left range, because
    // we go up to 255, not 256.
    float f = (value - p75) / (p100 - p75);
    ans = 192 + static_cast<int>(f * 63 + 0.5);
    if (ans < 192) ans = 192;
    if (ans > 255) ans = 255;
  }
  return static_cast<unsigned char>(ans);
}


// static
inline float CompressedMatrix::CharToFloat(
    float p0, float p25, float p75, float p100,
    unsigned char value) {
  if (value <= 64) {
    return p0 + (p25 - p0) * value * (1/64.0);
  } else if (value <= 192) {
    return p25 + (p75 - p25) * (value - 64) * (1/128.0);
  } else {
    return p75 + (p100 - p75) * (value - 192) * (1/63.0);
  }
}


template<class Real> // static
void CompressedMatrix::CompressColumn(
    const GlobalHeader &global_header,
    const Real *data, MatrixIndexT stride,
    int32 num_rows, CompressedMatrix::PerColHeader *header,
    unsigned char *byte_data) {
  ComputeColHeader(global_header, data, stride,
                   num_rows, header);

  float p0 = Uint16ToFloat(global_header, header->percentile_0),
      p25 = Uint16ToFloat(global_header, header->percentile_25),
      p75 = Uint16ToFloat(global_header, header->percentile_75),
      p100 = Uint16ToFloat(global_header, header->percentile_100);
  
  for (int32 i = 0; i < num_rows; i++) {
    Real this_data = data[i * stride];
    byte_data[i] = FloatToChar(p0, p25, p75, p100, this_data);
  }    
}

// static
unsigned char* CompressedMatrix::AllocateData(int32 num_bytes) {
  KALDI_ASSERT(num_bytes > 0);
  KALDI_COMPILE_TIME_ASSERT(sizeof(float) == 4);
  // round size up to nearest number of floats.
  return reinterpret_cast<unsigned char*>(new float[(num_bytes/3) + 4]);
}

void CompressedMatrix::Write(std::ostream &os, bool binary) const {
  if (binary) { // Binary-mode write:
    if (data_ != NULL) {
      GlobalHeader &h = *reinterpret_cast<GlobalHeader*>(data_);
      MatrixIndexT size = DataSize(h); // total size of data in data_
      os.write(reinterpret_cast<const char*>(data_), size);
    } else { // special case: where data_ == NULL, we treat it as an empty
      // matrix.
      GlobalHeader h;
      h.range = h.min_value = 0.0;
      h.num_rows = h.num_cols = 0;
      os.write(reinterpret_cast<const char*>(&h), sizeof(h));
    }
  } else {
    // Text-mode writing.  Only really useful for debug, but we'll implement it.
    if (data_ == NULL) {
      os << 0.0 << ' ' << 0.0 << ' ' << 0 << ' ' << 0 << '\n';
    } else {
      GlobalHeader &h = *reinterpret_cast<GlobalHeader*>(data_);
      KALDI_ASSERT(h.num_cols != 0);
      os << h.min_value << ' ' << h.range << ' ' << h.num_rows << ' ' << h.num_cols << '\n';
    
      PerColHeader *per_col_header = reinterpret_cast<PerColHeader*>(&h + 1);
      unsigned char *c = reinterpret_cast<unsigned char*>(per_col_header + h.num_cols);

      for (int32 i = 0; i < h.num_cols; i++, per_col_header++) {
        os << per_col_header->percentile_0 << ' ' << per_col_header->percentile_25
           << ' ' << per_col_header->percentile_75 
           << ' ' << per_col_header->percentile_100 << '\n';
        for (int32 j = 0; j < h.num_rows; j++, c++)
          os << static_cast<int>(*c) << ' ';
        os << '\n';
      }
    }
  }
  if (os.fail())
    KALDI_ERR << "Error writing compressed matrix to stream.";
}

void CompressedMatrix::Read(std::istream &is, bool binary) {
  if (data_ != NULL) {
    delete[] data_;
    data_ = NULL;
  }
  if (binary) { // Binary-mode read.
    GlobalHeader h;
    is.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (is.fail())
      KALDI_ERR << "Failed to read header";
    if (h.num_cols == 0) { // empty matrix.
      return;
    }
    int32 size = DataSize(h), remaining_size = size - sizeof(GlobalHeader);
    data_ = AllocateData(size);
    *(reinterpret_cast<GlobalHeader*>(data_)) = h;
    is.read(reinterpret_cast<char*>(data_ + sizeof(GlobalHeader)),
            remaining_size);
  } else { // Text-mode read.
    GlobalHeader h;
    is >> h.min_value >> h.range >> h.num_rows >> h.num_cols;
    if (is.fail())
      KALDI_ERR << "Failed to read header.";
    if (h.num_cols == 0) { // Empty matrix; null data_ pointer.
      return;
    }
    int32 size = DataSize(h);
    data_ = AllocateData(size);
    *(reinterpret_cast<GlobalHeader*>(data_)) = h;
    
    PerColHeader *per_col_header =
        reinterpret_cast<PerColHeader*>(data_ + sizeof(GlobalHeader));
    unsigned char *c =
        reinterpret_cast<unsigned char*>(per_col_header + h.num_cols);
    for (int32 i = 0; i < h.num_cols; i++, per_col_header++) {
      is >> per_col_header->percentile_0 >> per_col_header->percentile_25
         >> per_col_header->percentile_75 >> per_col_header->percentile_100;
      for (int32 j = 0; j < h.num_rows; j++, c++) {
        int i;
        is >> i;
        assert(i >= 0 && i <= 255);
        *c = static_cast<unsigned char>(i);
      }
    }
  }   
  if (is.fail())
    KALDI_ERR << "Failed to read data.";
}

template<class Real>
void CompressedMatrix::CopyToMat(Matrix<Real> *mat) const {
  if (data_ == NULL) {
    mat->Resize(0, 0);
  } else {
    GlobalHeader *h = reinterpret_cast<GlobalHeader*>(data_);
    PerColHeader *per_col_header = reinterpret_cast<PerColHeader*>(h+1);
    unsigned char *byte_data = reinterpret_cast<unsigned char*>(per_col_header +
                                                                h->num_cols);
    int32 num_cols = h->num_cols, num_rows = h->num_rows;
    mat->Resize(num_rows, num_cols, kUndefined);
    for (int32 i = 0; i < num_cols; i++, per_col_header++) {
      float p0 = Uint16ToFloat(*h, per_col_header->percentile_0),
          p25 = Uint16ToFloat(*h, per_col_header->percentile_25),
          p75 = Uint16ToFloat(*h, per_col_header->percentile_75),
          p100 = Uint16ToFloat(*h, per_col_header->percentile_100);
      for (int32 j = 0; j < num_rows; j++, byte_data++) {
        float f = CharToFloat(p0, p25, p75, p100, *byte_data);
        (*mat)(j, i) = f;
      }
    }
  }
}

// Instantiate the template for float and double.
template
void CompressedMatrix::CopyToMat(Matrix<float> *mat) const;
template
void CompressedMatrix::CopyToMat(Matrix<double> *mat) const;





} // namespace kaldi

