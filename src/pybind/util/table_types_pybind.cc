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

#include "util/table-types.h"

#include "util/kaldi-table-inl.h"

using namespace kaldi;

namespace {

template <class Holder>
void sequential_matrix_reader(py::module& m, const std::string& data_type) {
  std::string name = "SequentialBaseFloat" + data_type + "Reader";
  py::class_<SequentialTableReader<Holder>>(m, name.c_str())
      .def(py::init<>())
      .def(py::init<const std::string&>())
      .def("Open", &SequentialTableReader<Holder>::Open)
      .def("Done", &SequentialTableReader<Holder>::Done)
      .def("Key", &SequentialTableReader<Holder>::Key)
      .def("FreeCurrent", &SequentialTableReader<Holder>::FreeCurrent)
      .def("Value", &SequentialTableReader<Holder>::Value)
      .def("Next", &SequentialTableReader<Holder>::Next)
      .def("IsOpen", &SequentialTableReader<Holder>::IsOpen)
      .def("Close", &SequentialTableReader<Holder>::Close);
}

template <class Holder>
void random_access_matrix_reader(py::module& m, const std::string& data_type) {
  std::string name = "RandomAccessBaseFloat" + data_type + "Reader";
  py::class_<RandomAccessTableReader<Holder>>(m, name.c_str())
      .def(py::init<>())
      .def(py::init<const std::string&>())
      .def("Open", &RandomAccessTableReader<Holder>::Open)
      .def("IsOpen", &RandomAccessTableReader<Holder>::IsOpen)
      .def("Close", &RandomAccessTableReader<Holder>::Close)
      .def("HasKey", &RandomAccessTableReader<Holder>::HasKey)
      .def("Value", &RandomAccessTableReader<Holder>::Value);
}

template <class Holder>
void matrix_writer(py::module& m, const std::string& data_type) {
  std::string name = "BaseFloat" + data_type + "Writer";
  py::class_<TableWriter<Holder>>(m, name.c_str())
      .def(py::init<const std::string&>())
      .def("IsOpen", &TableWriter<Holder>::IsOpen)
      .def("Open", &TableWriter<Holder>::Open)
      .def("Write", &TableWriter<Holder>::Write)
      .def("Flush", &TableWriter<Holder>::Flush)
      .def("Close", &TableWriter<Holder>::Close);
}

}  // namespace

void pybind_table_types(py::module& m) {
  sequential_matrix_reader<KaldiObjectHolder<Matrix<float>>>(m, "Matrix");
  sequential_matrix_reader<KaldiObjectHolder<Vector<float>>>(m, "Vector");

  random_access_matrix_reader<KaldiObjectHolder<Matrix<float>>>(m, "Matrix");
  random_access_matrix_reader<KaldiObjectHolder<Vector<float>>>(m, "Vector");

  matrix_writer<KaldiObjectHolder<Matrix<float>>>(m, "Matrix");
  matrix_writer<KaldiObjectHolder<Vector<float>>>(m, "Vector");
}
