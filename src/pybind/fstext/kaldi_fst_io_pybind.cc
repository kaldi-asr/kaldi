// pybind/fstext/kaldi_fst_io_pybind.cc

// Copyright 2019   Mobvoi AI Lab, Beijing, China
//                  (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

// See ../../../COPYING for clarification regarding multiple authors
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

#include "fstext/kaldi_fst_io_pybind.h"

#include "fstext/kaldi-fst-io.h"

void pybind_kaldi_fst_io(py::module& m) {
  m.def("ReadFstKaldi", (fst::StdVectorFst * (*)(std::string))fst::ReadFstKaldi,
        "Read a binary FST using Kaldi I/O mechanisms (pipes, etc.) On error, "
        "throws using KALDI_ERR.  Note: this doesn't support the text-mode "
        "option that we generally like to support.",
        py::arg("rxfilename"), py::return_value_policy::reference);

  m.def("ReadFstKaldiGeneric", fst::ReadFstKaldiGeneric,
        "Read a binary FST using Kaldi I/O mechanisms (pipes, etc.) If it "
        "can't read the FST, if throw_on_err == true it throws using "
        "KALDI_ERR; otherwise it prints a warning and returns. Note:this "
        "doesn't support the text-mode option that we generally like to "
        "support. This version currently supports ConstFst<StdArc> or "
        "VectorFst<StdArc> (const-fst can give better performance for "
        "decoding).",
        py::arg("rxfilename"), py::arg("throw_on_err") = true,
        py::return_value_policy::reference);

  m.def("CastOrConvertToVectorFst", &fst::CastOrConvertToVectorFst,
        "This function attempts to dynamic_cast the pointer 'fst' (which will "
        "likely have been returned by ReadFstGeneric()), to the more derived "
        "type VectorFst<StdArc>. If this succeeds, it returns the same "
        "pointer; if it fails, it converts the FST type (by creating a new "
        "VectorFst<stdArc> initialized by 'fst'), prints a warning, and "
        "deletes 'fst'.",
        py::arg("fst"), py::return_value_policy::reference);

  m.def("ReadFstKaldi",
        (void (*)(std::string, fst::StdVectorFst*)) & fst::ReadFstKaldi,
        "Version of ReadFstKaldi() that writes to a pointer.  Assumes the FST "
        "is binary with no binary marker.  Crashes on error.",
        py::arg("rxfilename"), py::arg("ofst"));

  m.def("WriteFstKaldi",
        (void (*)(const fst::StdVectorFst&, std::string)) & fst::WriteFstKaldi,
        "Write an FST using Kaldi I/O mechanisms (pipes, etc.) On error, "
        "throws using KALDI_ERR.  For use only in code in fstbin/, as it "
        "doesn't support the text-mode option.",
        py::arg("fst"), py::arg("wxfilename"));

  m.def("WriteFstKaldi",
        (void (*)(std::ostream&, bool, const fst::StdVectorFst&)) &
            fst::WriteFstKaldi,
        "This is a more general Kaldi-type-IO mechanism of writing FSTs to "
        "streams, supporting binary or text-mode writing.  (note: we just "
        "write the integers, symbol tables are not supported). On error, "
        "throws using KALDI_ERR.",
        py::arg("os"), py::arg("binary"), py::arg("fst"));

  m.def("ReadFstKaldi",
        (void (*)(std::istream&, bool, fst::StdVectorFst*)) & fst::ReadFstKaldi,
        "A generic Kaldi-type-IO mechanism of reading FSTs from streams, "
        "supporting binary or text-mode reading/writing.",
        py::arg("is"), py::arg("binary"), py::arg("fst"));
  m.def("ReadAndPrepareLmFst", &fst::ReadAndPrepareLmFst,
        "Read an FST file for LM (G.fst) and make it an acceptor, and make "
        "sure it is sorted on labels",
        py::arg("rxfilename"), py::return_value_policy::reference);

  {
    // fangjun: it should be called StdVectorFstHolder to match the naming
    // convention in OpenFst but kaldi uses only StdArc so there is no confusion
    // here.
    using PyClass = fst::VectorFstHolder;
    py::class_<PyClass>(m, "VectorFstHolder")
        .def(py::init<>())
        .def_static("Write", &PyClass::Write, py::arg("os"), py::arg("binary"),
                    py::arg("t"))
        .def("Copy", &PyClass::Copy)
        .def("Read", &PyClass::Read, "Reads into the holder.", py::arg("is"));
  }
}
