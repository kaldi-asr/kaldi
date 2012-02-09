// matrix/kaldi-vector-inl.h

// Copyright 2009-2011   Ondrej Glembek;  Microsoft Corporation

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

// This is an internal header file, included by other library headers.
// You should not attempt to use it directly.

#ifndef KALDI_MATRIX_KALDI_VECTOR_INL_H_
#define KALDI_MATRIX_KALDI_VECTOR_INL_H_ 1

namespace kaldi {

template<typename Real>
std::ostream & operator << (std::ostream& os, const VectorBase<Real>& rv) {
  rv.Write(os, false);
  return os;
}

template<typename Real>
std::istream &operator >> (std::istream& is, VectorBase<Real>& rv) {
  rv.Read(is, false);
  return is;
}

template<typename Real>
std::istream &operator >> (std::istream& is, Vector<Real>& rv) {
  rv.Read(is, false);
  return is;
}

template<>
float VecVec<>(const VectorBase<float>& ra, const VectorBase<float>& rb);

template<>
double VecVec<>(const VectorBase<double>& ra, const VectorBase<double>& rb);

template<>
template<>
void VectorBase<float>::AddVec(const float alpha, const VectorBase<float>& rv);

template<>
template<>
void VectorBase<double>::AddVec<double>(const double alpha,
                                        const VectorBase<double>& rv);

template<>
void VectorBase<float>::AddMatVec(const float alpha, const MatrixBase<float>& M,
                                  MatrixTransposeType trans,
                                  const VectorBase<float>& v, const float beta);

template<>
void VectorBase<double>::AddMatVec(const double alpha,
                                   const MatrixBase<double>& M,
                                   MatrixTransposeType trans,
                                   const VectorBase<double>& v,
                                   const double beta);

template<>
void VectorBase<float>::AddSpVec(const float alpha, const SpMatrix<float>& M,
                                 const VectorBase<float>& v, const float beta);

template<>
void VectorBase<double>::AddSpVec(const double alpha, const SpMatrix<double>& M,
                                  const VectorBase<double>& v,
                                  const double beta);

template<>
void VectorBase<double>::Scale(double alpha);

template<>
void VectorBase<float>::Scale(float alpha);

}  // namespace kaldi

#endif  // KALDI_MATRIX_KALDI_VECTOR_INL_H_
