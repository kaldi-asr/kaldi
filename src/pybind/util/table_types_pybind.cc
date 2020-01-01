// pybind/util/table_types_pybind.cc

// Copyright 2019   Daniel Povey
//           2019   Dongji Gao
//           2019   Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)

// See ../../../COPYING for clarification regarding multiple authors
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

#include "util/table_types_pybind.h"

#include "util/kaldi_table_pybind.h"

#include "util/kaldi-table-inl.h"
#include "util/table-types.h"

using namespace kaldi;

void pybind_table_types(py::module& m) {
  pybind_sequential_table_reader<KaldiObjectHolder<Matrix<float>>>(
      m, "_SequentialBaseFloatMatrixReader");

  pybind_random_access_table_reader<KaldiObjectHolder<Matrix<float>>>(
      m, "_RandomAccessBaseFloatMatrixReader");

  pybind_table_writer<KaldiObjectHolder<Matrix<float>>>(
      m, "_BaseFloatMatrixWriter");

  pybind_sequential_table_reader<KaldiObjectHolder<Vector<float>>>(
      m, "_SequentialBaseFloatVectorReader");

  pybind_random_access_table_reader<KaldiObjectHolder<Vector<float>>>(
      m, "_RandomAccessBaseFloatVectorReader");

  pybind_table_writer<KaldiObjectHolder<Vector<float>>>(
      m, "_BaseFloatVectorWriter");

  pybind_table_writer<KaldiObjectHolder<CompressedMatrix>>(
      m, "_CompressedMatrixWriter");
}
