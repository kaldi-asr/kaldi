// pybind/kaldi_pybind.cc

// Copyright 2019   Daniel Povey
//           2019   Dongji Gao

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include "matrix/kaldi-matrix.h"
#include "util/table-types.h"
#include "util/kaldi-table-inl.h"

using namespace kaldi;

template<class Holder>
void sequential_matrix_reader(py::module &m, const std::string &data_type) {
    std::string name = "SequentialBaseFloat" + data_type + "Reader";
    py::class_<SequentialTableReader<Holder>>(m, name.c_str())
        .def(py::init<>()) 
        .def(py::init<const std::string &>()) 
        .def("Open", &SequentialTableReader<Holder>::Open) 
        .def("Done", &SequentialTableReader<Holder>::Done) 
        .def("Key", &SequentialTableReader<Holder>::Key)   
        .def("FreeCurrent", &SequentialTableReader<Holder>::FreeCurrent) 
        .def("Value", &SequentialTableReader<Holder>::Value) 
        .def("Next", &SequentialTableReader<Holder>::Next) 
        .def("IsOpen", &SequentialTableReader<Holder>::IsOpen) 
        .def("Close", &SequentialTableReader<Holder>::Close);  
}

template<class Holder>
void random_access_matrix_reader(py::module &m, const std::string &data_type) {
    std::string name = "RandomAccessBaseFloat" + data_type + "Reader";
    py::class_<RandomAccessTableReader<Holder>>(m, name.c_str())
        .def(py::init<>()) 
        .def(py::init<const std::string &>()) 
        .def("Open", &RandomAccessTableReader<Holder>::Open) 
        .def("IsOpen", &RandomAccessTableReader<Holder>::IsOpen) 
        .def("Close", &RandomAccessTableReader<Holder>::Close)
        .def("HasKey", &RandomAccessTableReader<Holder>::HasKey)
        .def("Value", &RandomAccessTableReader<Holder>::Value);  
}

template<class Holder>
void matrix_writer(py::module &m, const std::string &data_type) {
  std::string name = "BaseFloat" + data_type + "Writer";
  py::class_<TableWriter<Holder>>(m, name.c_str())
      .def(py::init<const std::string &>())
      .def("IsOpen", &TableWriter<Holder>::IsOpen)
      .def("Open", &TableWriter<Holder>::Open)    
      .def("Write", &TableWriter<Holder>::Write)  
      .def("Flush", &TableWriter<Holder>::Flush)  
      .def("Close", &TableWriter<Holder>::Close); 
}

PYBIND11_MODULE(kaldi_pybind, m) {
  m.doc() = "pybind11 binding of some things from kaldi's src/matrix directory. "
      "Source is in $(KALDI_ROOT)/src/pybind/matrix-lib.cc";

  py::enum_<MatrixResizeType>(m, "MatrixResizeType", py::arithmetic(), "Matrix initialization policies")
      .value("kSetZero", kSetZero, "Set to zero")
      .value("kUndefined", kUndefined, "Leave undefined")
      .value("kCopyData", kCopyData, "Copy any previously existing data")
      .export_values();

  py::enum_<MatrixStrideType>(m, "MatrixStrideType", py::arithmetic(), "Matrix stride policies")
      .value("kDefaultStride", kDefaultStride, "Set to a multiple of 16 in bytes")
      .value("kStrideEqualNumCols", kStrideEqualNumCols, "Set to the number of columns")
      .export_values();

  py::class_<Vector<float> >(m, "FloatVector", pybind11::buffer_protocol())
      .def_buffer([](const Vector<float> &v) -> pybind11::buffer_info {
    return pybind11::buffer_info(
        (void*)v.Data(),
        sizeof(float),
        pybind11::format_descriptor<float>::format(),
        1, // num-axes
        { v.Dim() },
        { 4 }); // strides (in chars)
        })
      .def("Dim", &Vector<float>::Dim, "Return the dimension of the vector")
      .def("__repr__",
           [] (const Vector<float> &a) -> std::string {
             std::ostringstream str;  a.Write(str, false); return str.str();
           })
      .def(py::init<const MatrixIndexT, MatrixResizeType>(),
           py::arg("size"), py::arg("resize_type") = kSetZero);

  py::class_<Matrix<float> >(m, "FloatMatrix", pybind11::buffer_protocol())
      .def_buffer([](const Matrix<float> &m) -> pybind11::buffer_info {
    return pybind11::buffer_info(
        (void*)m.Data(), // pointer to buffer
        sizeof(float),   // size of one scalar 
        pybind11::format_descriptor<float>::format(),
        2,               // num-axes
        { m.NumRows(), m.NumCols() },    // buffer dimensions
        { sizeof(float) * m.Stride(), sizeof(float) });  // stride for each index (in chars)
        })
      .def("NumRows", &Matrix<float>::NumRows, "Return number of rows")
      .def("NumCols", &Matrix<float>::NumCols, "Return number of columns")
      .def("Stride", &Matrix<float>::Stride, "Return stride")
      .def("__repr__",
           [] (const Matrix<float> &b) -> std::string {
             std::ostringstream str; b.Write(str, false); return str.str();
           })
      .def(py::init<const MatrixIndexT, const MatrixIndexT,
          MatrixResizeType, MatrixStrideType>(),
          py::arg("row"), py::arg("col"), py::arg("resize_type") = kSetZero,
          py::arg("stride_type") = kDefaultStride);


  sequential_matrix_reader<KaldiObjectHolder<Matrix<float>>>(m, "Matrix");
  sequential_matrix_reader<KaldiObjectHolder<Vector<float>>>(m, "Vector");

  random_access_matrix_reader<KaldiObjectHolder<Matrix<float>>>(m, "Matrix");
  random_access_matrix_reader<KaldiObjectHolder<Vector<float>>>(m, "Vector");

  matrix_writer<KaldiObjectHolder<Matrix<float>>>(m, "Matrix");
  matrix_writer<KaldiObjectHolder<Vector<float>>>(m, "Vector");
}

