// matrix/packed-matrix-inl.h
// Copyright 2009-2011  Ondrej Glembek  Microsoft Corporation  Lukas Burget  Arnab Ghoshal
//   Yanmin Qian  Jan Silovsky

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
#ifndef KALDI_MATRIX_PACKED_MATRIX_INL_H_
#define KALDI_MATRIX_PACKED_MATRIX_INL_H_

namespace kaldi {

template<>
void PackedMatrix<float>::Scale(float alpha);

template<>
void PackedMatrix<double>::Scale(double alpha);

}  // namespace kaldi

#endif

