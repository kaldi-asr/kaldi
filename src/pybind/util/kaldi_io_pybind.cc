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
        .def(py::init<const std::string&, bool*>(),
             "The normal constructor.  Opens the stream in binary mode. "
             "Equivalent to calling the default constructor followed by "
             "Open(); then, if binary != NULL, it calls ReadHeader(), putting "
             "the output in 'binary'; it throws on error.",
             py::arg("rxfilename"), py::arg("contents_binary") = nullptr)
        .def("Open", &PyClass::Open,
             "Open opens the stream for reading (the mode, where relevant, is "
             "binary; use OpenTextMode for text-mode, we made this a separate "
             "function rather than a boolean argument, to avoid confusion with "
             "Kaldi's text/binary distinction, since reading in the file "
             "system's text mode is unusual.)  If contents_binary != NULL, it "
             "reads the binary-mode header and puts it in the  'binary' "
             "variable.  Returns true on success.  If it returns false it will "
             "not be open.  You may call Open even if it is already open; it "
             "will close the existing stream and reopen (however if closing "
             "the old stream failed it will throw).",
             py::arg("rxfilename"), py::arg("contents_binary") = nullptr)
        .def("OpenTextMode", &PyClass::OpenTextMode,
             "As Open but (if the file system has text/binary modes) opens in "
             "text mode; you shouldn't ever have to use this as in Kaldi we "
             "read even text files in binary mode (and ignore the \r).",
             py::arg("rxfilename"))
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
