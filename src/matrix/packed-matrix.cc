// matrix/packed-matrix.cc

// Copyright 2009-2012  Microsoft Corporation  Saarland University
//        Johns Hopkins University (Author: Daniel Povey)

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
/**
 * @file packed-matrix.cc
 *
 * Implementation of specialized PackedMatrix template methods
 */

#include "matrix/packed-matrix.h"
#include "matrix/kaldi-vector.h"

namespace kaldi {

template<>
void PackedMatrix<float>::AddPacked(const float alpha,
                                    const PackedMatrix<float> &rMa) {
  KALDI_ASSERT(num_rows_ == rMa.NumRows());
  MatrixIndexT sz =  ((num_rows_ * (num_rows_+1)) / 2);
  cblas_saxpy(sz, alpha, rMa.Data(), 1, data_, 1);
}

template<>
void PackedMatrix<double>::AddPacked(const double alpha,
                                     const PackedMatrix<double> &rMa) {
  KALDI_ASSERT(num_rows_ == rMa.NumRows());
  MatrixIndexT sz = ((num_rows_ * (num_rows_+1)) / 2);
  cblas_daxpy(sz, alpha, rMa.Data(), 1, data_, 1);
}

template<typename Real>
inline void PackedMatrix<Real>::Init(MatrixIndexT r) {
  if (r == 0) {
    num_rows_ = 0;
    data_ = 0;
#ifdef KALDI_MEMALIGN_MANUAL
    free_data_=NULL;
#endif
    return;
  }
  MatrixIndexT size = r * (r + 1) / 2 * sizeof(Real);

  void *data;  // aligned memory block
  void* free_data;  // memory block to be really freed

  if ((data = KALDI_MEMALIGN(16, size, &free_data)) != NULL) {
    this->data_ = static_cast<Real *> (data);
#ifdef KALDI_MEMALIGN_MANUAL
    this->free_data_ = static_cast<Real *> (free_data);
#endif
    this->num_rows_ = r;
  } else {
    throw std::bad_alloc();
  }
}

template<typename Real>
void PackedMatrix<Real>::Swap(PackedMatrix<Real> *other) {
  std::swap(data_, other->data_);
  std::swap(num_rows_, other->num_rows_);
#ifdef KALDI_MEMALIGN_MANUAL
  std::swap(free_data_, other->free_data_);
#endif
}


template<typename Real>
void PackedMatrix<Real>::Resize(MatrixIndexT r, MatrixResizeType resize_type) {
  // the next block uses recursion to handle what we have to do if
  // resize_type == kCopyData.
  if (resize_type == kCopyData) {
    if (this->data_ == NULL || r == 0) resize_type = kSetZero;  // nothing to copy.
    else if (this->num_rows_ == r) { return; } // nothing to do.
    else {
      // set tmp to a packed matrix of the desired size.
      PackedMatrix<Real> tmp(r, kUndefined);
      MatrixIndexT r_min = std::min(r, num_rows_);
      size_t mem_size_min = sizeof(Real) * (r_min*(r_min+1))/2,
          mem_size_full = sizeof(Real) * (r*(r+1))/2;
      // Copy the contents to tmp.
      memcpy(tmp.data_, data_, mem_size_min);
      char *ptr = static_cast<char*>(static_cast<void*>(tmp.data_));
      // Set the rest of the contents of tmp to zero.
      memset(static_cast<void*>(ptr + mem_size_min), 0, mem_size_full-mem_size_min);
      tmp.Swap(this);
      return;
    }
  }
  if (data_ != NULL) Destroy();
  Init(r);
  if (resize_type == kSetZero) SetZero();
}



template<typename Real>
void PackedMatrix<Real>::ScaleDiag(Real alpha) {
  Real *ptr = data_;
  for (MatrixIndexT i = 2; i <= num_rows_+1; i++) {
    *ptr *= alpha;
    ptr += i;
  }
}


template<typename Real>
template<typename OtherReal>
void PackedMatrix<Real>::CopyFromPacked(const PackedMatrix<OtherReal> &orig) {
  KALDI_ASSERT(NumRows() == orig.NumRows());
  if (sizeof(Real) == sizeof(OtherReal)) {
    memcpy(data_, orig.Data(), SizeInBytes());
  } else {
    Real *dst = data_;
    const OtherReal *src = orig.Data();
    MatrixIndexT size = (NumRows() * (NumRows()+1) / 2);
    for (MatrixIndexT i = 0; i < size; i++, dst++, src++)
      *dst = *src;
  }
}

// template instantiations.
template
void PackedMatrix<float>::CopyFromPacked(const PackedMatrix<double> &orig);
template
void PackedMatrix<double>::CopyFromPacked(const PackedMatrix<float> &orig);
template
void PackedMatrix<double>::CopyFromPacked(const PackedMatrix<double> &orig);
template
void PackedMatrix<float>::CopyFromPacked(const PackedMatrix<float> &orig);



template<typename Real>
template<typename OtherReal>
void PackedMatrix<Real>::CopyFromVec(const SubVector<OtherReal> &vec) {
  MatrixIndexT size = (NumRows()*(NumRows()+1)) / 2;
  KALDI_ASSERT(vec.Dim() == size);
  if (sizeof(Real) == sizeof(OtherReal)) {
    memcpy(data_, vec.Data(), size * sizeof(Real));
  } else {
    Real *dst = data_;
    const OtherReal *src = vec.Data();
    for (MatrixIndexT i = 0; i < size; i++, dst++, src++)
      *dst = *src;
  }
}

// template instantiations.
template
void PackedMatrix<float>::CopyFromVec(const SubVector<double> &orig);
template
void PackedMatrix<double>::CopyFromVec(const SubVector<float> &orig);
template
void PackedMatrix<double>::CopyFromVec(const SubVector<double> &orig);
template
void PackedMatrix<float>::CopyFromVec(const SubVector<float> &orig);



template<typename Real>
void PackedMatrix<Real>::SetZero() {
  memset(data_, 0, SizeInBytes());
}

template<typename Real>
void PackedMatrix<Real>::SetUnit() {
  memset(data_, 0, SizeInBytes());
  for (MatrixIndexT row = 0;row < num_rows_;row++)
    (*this)(row, row) = 1.0;
}

template<typename Real>
Real PackedMatrix<Real>::Trace() const {
  Real ans = 0.0;
  for (MatrixIndexT row = 0;row < num_rows_;row++)
    ans += (*this)(row, row);
  return ans;
}

template<typename Real>
void PackedMatrix<Real>::Destroy() {
  // we need to free the data block if it was defined
#ifndef KALDI_MEMALIGN_MANUAL
  if (data_ != NULL) KALDI_MEMALIGN_FREE(data_);
#else
  if (free_data_ != NULL) KALDI_MEMALIGN_FREE(free_data_);
  free_data_ = NULL;
#endif

  data_ = NULL;
  num_rows_ = 0;
}

template<>
void PackedMatrix<float>::Scale(float alpha) {
  MatrixIndexT sz = ((num_rows_ * (num_rows_+1)) / 2);
  cblas_sscal(sz, alpha, data_, 1);
}

template<>
void PackedMatrix<double>::Scale(double alpha) {
  MatrixIndexT sz = ((num_rows_ * (num_rows_+1)) / 2);
  cblas_dscal(sz, alpha, data_, 1);
}

template<typename Real>
void PackedMatrix<Real>::Write(std::ostream &os, bool binary) const {
  if (!os.good()) {
    KALDI_ERR << "Failed to write vector to stream: stream not good";
  }
  std::string my_token = (sizeof(Real) == 4 ? "FP" : "DP");

  WriteToken(os, binary, my_token);

  int32 size = this->NumRows();  // make the size 32-bit on disk.
  KALDI_ASSERT(this->NumRows() == (MatrixIndexT) size);
  WriteBasicType(os, binary, size);

  // We don't use the built-in Kaldi write routines for the floats, as they are
  // not efficient enough.
  MatrixIndexT num_elems = ((size+1)*(MatrixIndexT)size)/2;
  if (!binary) {
    for (MatrixIndexT i = 0; i < num_elems; i++)
      WriteBasicType(os, binary, data_[i]);
    os << '\n';
  } else {
    os.write((const char*) data_, sizeof(Real) * num_elems);
  }
  if (os.fail()) {
    KALDI_ERR << "Failed to write packed matrix to stream";
  }
}


// template<typename Real>
//   void Save (std::ostream & os, const PackedMatrix<Real>& rM)
//   {
//     const Real* p_elem = rM.data();
//     for (MatrixIndexT i = 0; i < rM.NumRows(); i++) {
//       for (MatrixIndexT j = 0; j <= i ; j++) {
//         os << *p_elem;
//         p_elem++;
//         if (j == i) {
//           os << '\n';
//         }
//         else {
//           os << ' ';
//         }
//       }
//     }
//     if (os.fail())
//       KALDI_ERR("Failed to write packed matrix to stream");
//   }





template<typename Real>
void PackedMatrix<Real>::Read(std::istream & is, bool binary, bool add) {
  if (add) {
    PackedMatrix<Real> tmp;
    tmp.Read(is, binary, false);  // read without adding.
    if (this->NumRows() == 0) this->Resize(tmp.NumRows());
    else {
      if (this->NumRows() != tmp.NumRows()) {
        if (tmp.NumRows() == 0) return;  // do nothing in this case.
        else KALDI_ERR << "PackedMatrix::Read, size mismatch " << this->NumRows()
                       << " vs. " << tmp.NumRows();
      }
    }
    this->AddPacked(1.0, tmp);
    return;
  } // now assume add == false.

  std::ostringstream specific_error;
  MatrixIndexT pos_at_start = is.tellg();
  int peekval = Peek(is, binary);
  const char *my_token =  (sizeof(Real) == 4 ? "FP" : "DP");
  char other_token_start = (sizeof(Real) == 4 ? 'D' : 'F');
  int32 size;
  MatrixIndexT num_elems;

  if (peekval == other_token_start) {  // need to instantiate the other type to read it.
    typedef typename OtherReal<Real>::Real OtherType;  // if Real == float, OtherType == double, and vice versa.
    PackedMatrix<OtherType> other(this->NumRows());
    other.Read(is, binary, false);  // add is false at this point.
    this->Resize(other.NumRows());
    this->CopyFromPacked(other);
    return;
  }
  std::string token;
  ReadToken(is, binary, &token);
  if (token != my_token) {
    specific_error << ": Expected token " << my_token << ", got " << token;
    goto bad;
  }
  ReadBasicType(is, binary, &size);  // throws on error.
  if ((MatrixIndexT)size != this->NumRows()) {
    KALDI_ASSERT(size>=0);
    this->Resize(size);
  }
  num_elems = ((size+1)*(MatrixIndexT)size)/2;
  if (!binary) {
    for (MatrixIndexT i = 0; i < num_elems; i++) {
      ReadBasicType(is, false, data_+i);  // will throw on error.
    }
  } else {
    if (num_elems)
      is.read(reinterpret_cast<char*>(data_), sizeof(Real)*num_elems);
  }
  if (is.fail()) goto bad;
  return;
bad:
  KALDI_ERR << "Failed to read packed matrix from stream. " << specific_error
            << " File position at start is "
            << pos_at_start << ", currently " << is.tellg();
}



template<>
void PackedMatrix<float>::AddVec2(const float alpha, const Vector<float> &rv) {
  KALDI_ASSERT(rv.Dim() == num_rows_);
  cblas_sspr(CblasRowMajor, CblasLower, rv.Dim(), alpha, rv.Data(), 1, data_);
}

template<>
void PackedMatrix<double>::AddVec2(const double alpha, const Vector<double> &rv) {
  KALDI_ASSERT(rv.Dim() == num_rows_);
  cblas_dspr(CblasRowMajor, CblasLower, rv.Dim(), alpha, rv.Data(), 1, data_);
}


// Instantiate PackedMatrix for float and double.
template
class PackedMatrix<float>;

template
class PackedMatrix<double>;


}  // namespace kaldi

