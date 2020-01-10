// pybind/util/kaldi_io_pybind.cc

// Copyright 2019   Mobvoi AI Lab, Beijing, China
//                  (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

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

#include "util/kaldi_io_pybind.h"

#include "util/kaldi-io.h"

using namespace kaldi;

void pybind_kaldi_io(py::module& m) {
  py::class_<std::istream>(m, "istream");
  {
    using PyClass = Input;
    py::class_<PyClass>(m, "Input")
        .def(py::init<>())
        // the constructor and `Open` both require a `bool*` argument
        // but pybind11 does not support passing a pointer to a primitive
        // type, only pointer to customized type is allowed.
        //
        // For more information, please refer to
        // https://github.com/pybind/pybind11/pull/1760/commits/1d8caa5fbd0903cece06ae646447fff9b4aa33c0
        // https://github.com/pybind/pybind11/pull/1760
        //
        // Were it be `bool*`, would it always be non-NULL in C++!
        .def(
            "Open",
            [](PyClass* ki, const std::string& rxfilename,
               bool read_header = false) -> std::vector<int> {
              // WARNING(fangjun): we cannot use `std::vector<bool> res;` here
              // since it is invalid to use `&result[0]` if it is a bool vector
              std::vector<int> result(1, 0);
              if (read_header) {
                result.resize(2);
                bool tmp;
                result[0] = ki->Open(rxfilename, &tmp);
                result[1] = tmp;
              } else {
                result[0] = ki->Open(rxfilename);
              }
              return result;
            },
            "Open the stream for reading. "
            "if `read_header` is true, then calls `ReadHeader()`, putting the "
            "output in the 1st position of the return value (counting from 0); "
            "if `read_header is false, the return value has only one element`. "
            "The return value has two elements if `read_header` is true, else "
            " contains only one element. The zeroth element indicates whether "
            "`Open` succeeds or not.",
            py::arg("rxfilename"), py::arg("read_header") = false)
        .def("IsOpen", &PyClass::IsOpen,
             "Return true if currently open for reading and Stream() will "
             "succeed.  Does not guarantee that the stream is good.")
        .def("Close", &PyClass::Close,
             "It is never necessary or helpful to call Close, except if you "
             "are concerned about to many filehandles being open. Close does "
             "not throw. It returns the exit code as int32 in the case of a "
             "pipe [kPipeInput], and always zero otherwise.")
        .def("Stream", &PyClass::Stream,
             "Returns the underlying stream. Throws if !IsOpen()",
             py::return_value_policy::reference);
  }
}
