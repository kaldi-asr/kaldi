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
        .def("Open",
             [](PyClass* ki, const std::string& rxfilename,
                bool read_header = false) -> std::vector<bool> {
               std::vector<bool> result(1, false);
               if (read_header) {
                 result.resize(2, false);
                 bool tmp;
                 result[0] = ki->Open(rxfilename, &tmp);
                 result[1] = tmp;
               } else {
                 result[0] = ki->Open(rxfilename);
               }
               return result;
             },
             "Open the stream for reading. "
             "Return a vector containing one bool or two depending on "
             "whether `read_header` is false or true."
             "\n",
             "(1) If `read_header` is true, it returns [opened, binary], where "
             "`opened` is true if the stream was opened successfully, false "
             "otherwise;\n"
             "`binary` is true if the stream was opened **and** in binary "
             "format\n"
             "\n"
             "(2) If `read_header` is false, it returns [opened], where "
             "`opened` is true if the stream was opened successfully, false "
             "otherwise",
             py::arg("rxfilename"), py::arg("read_header") = false)
        // the constructor and `Open` method both require a `bool*` argument
        // but pybind11 does not support passing a pointer to a primitive
        // type, only pointer to customized type is allowed.
        //
        // For more information, please refer to
        // https://github.com/pybind/pybind11/pull/1760/commits/1d8caa5fbd0903cece06ae646447fff9b4aa33c0
        // https://github.com/pybind/pybind11/pull/1760
        //
        // Was it a `bool*`, would it always be non-NULL in C++!
        //
        // Therefore, we wrap the `Open` method and do NOT wrap the
        // `constructor` with `bool*` arguments
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

  py::class_<std::ostream>(m, "ostream");
  {
    using PyClass = Output;
    py::class_<PyClass>(m, "Output")
        .def(py::init<>())
        .def(py::init<const std::string&, bool, bool>(),
             "The normal constructor, provided for convenience. Equivalent to "
             "calling with default constructor then Open() with these "
             "arguments.",
             py::arg("filename"), py::arg("binary"),
             py::arg("write_header") = true)
        .def("Open", &PyClass::Open,
             "This opens the stream, with the given mode (binary or text).  It "
             "returns true on success and false on failure.  However, it will "
             "throw if something was already open and could not be closed (to "
             "avoid this, call Close() first.  if write_header == true and "
             "binary == true, it writes the Kaldi binary-mode header ('\0' "
             "then 'B').  You may call Open even if it is already open; it "
             "will close the existing stream and reopen (however if closing "
             "the old stream failed it will throw).",
             py::arg("wxfilename"), py::arg("binary"), py::arg("write_header"))
        .def("IsOpen", &PyClass::IsOpen,
             "return true if we have an open "
             "stream.  Does not imply stream is "
             "good for writing.")
        .def("Stream", &PyClass::Stream,
             "will throw if not open; else returns stream.",
             py::return_value_policy::reference)
        .def("Close", &PyClass::Close,
             "Close closes the stream. Calling Close is never necessary unless "
             "you want to avoid exceptions being thrown.  There are times when "
             "calling Close will hurt efficiency (basically, when using "
             "offsets into files, and using the same Input object), but most "
             "of the time the user won't be doing this directly, it will be "
             "done in kaldi-table.{h, cc}, so you don't have to worry about "
             "it.");
  }
}
