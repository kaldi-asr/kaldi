// pybind/fst/compile_pybind.cc

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

#include "fst/compile_pybind.h"

#include "fst/script/compile.h"

void pybind_compile(py::module& m) {
  m.def(
      "CompileFst",
      [](std::string& text_fst_str, const std::string& out_binary_fst_filename,
         const std::string& source = "standard_input",
         const string& fst_type = "vector", const string& arc_type = "standard",
         const fst::SymbolTable* isyms = nullptr,
         const fst::SymbolTable* osyms = nullptr,
         const fst::SymbolTable* ssyms = nullptr, bool accep = false,
         bool ikeep = false, bool okeep = false, bool nkeep = false,
         bool allow_negative_labels = false) {
        // (fangjun): paramemter `source` is only for debugging !
        std::stringstream strm;
        strm << text_fst_str;
        fst::script::CompileFst(strm, source, out_binary_fst_filename, fst_type,
                                arc_type, isyms, osyms, ssyms, accep, ikeep,
                                okeep, nkeep, allow_negative_labels);
      },
      "the fst is written to out_binary_fst_filename", py::arg("text_fst_str"),
      py::arg("out_binary_fst_filename"), py::arg("source") = "standard input",
      py::arg("fst_type") = "vector", py::arg("arc_type") = "standard",
      py::arg("isymbols") = nullptr, py::arg("osymbols") = nullptr,
      py::arg("ssymbols") = nullptr, py::arg("acceptor") = false,
      py::arg("keep_isymbols") = false, py::arg("keep_osymbols") = false,
      py::arg("keep_state_numbering") = false,
      py::arg("allow_negative_labels") = false);
}
