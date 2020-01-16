// pybind/lat/determinize_lattice_pruned_pybind.cc

// Copyright 2020   Mobvoi AI Lab, Beijing, China
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

#include "lat/determinize_lattice_pruned_pybind.h"

#include "lat/determinize-lattice-pruned.h"

void pybind_determinize_lattice_pruned(py::module& m) {
  {
    using PyClass = fst::DeterminizeLatticePrunedOptions;

    py::class_<PyClass>(m, "DeterminizeLatticePrunedOptions")
        .def(py::init<>())
        .def_readwrite("delta", &PyClass::delta,
                       "A small offset used to measure equality of weights.")
        .def_readwrite("max_mem", &PyClass::max_mem,
                       "If >0, determinization will fail and return false when "
                       "the algorithm's (approximate) memory consumption "
                       "crosses this threshold.")
        .def_readwrite("max_loop", &PyClass::max_loop,
                       "If >0, can be used to detect non-determinizable input "
                       "(a case that wouldn't be caught by max_mem).")
        .def_readwrite("max_states", &PyClass::max_states)
        .def_readwrite("max_arcs", &PyClass::max_arcs)
        .def_readwrite("retry_cutoff", &PyClass::retry_cutoff)
        .def("__str__", [](const PyClass& opt) {
          std::ostringstream os;
          os << "delta: " << opt.delta << "\n";
          os << "max_mem: " << opt.max_mem << "\n";
          os << "max_loop: " << opt.max_loop << "\n";
          os << "max_states: " << opt.max_states << "\n";
          os << "max_arcs: " << opt.max_arcs << "\n";
          os << "retry_cutoff: " << opt.retry_cutoff << "\n";
          return os.str();
        });
  }

  {
    using PyClass = fst::DeterminizeLatticePhonePrunedOptions;

    py::class_<PyClass>(m, "DeterminizeLatticePhonePrunedOptions")
        .def(py::init<>())
        .def_readwrite("delta", &PyClass::delta,
                       "A small offset used to measure equality of weights.")
        .def_readwrite("max_mem", &PyClass::max_mem,
                       "If >0, determinization will fail and return false when "
                       "the algorithm's (approximate) memory consumption "
                       "crosses this threshold.")
        .def_readwrite("phone_determinize", &PyClass::phone_determinize,
                       "phone_determinize: if true, do a first pass "
                       "determinization on both phones and words.")
        .def_readwrite("word_determinize", &PyClass::word_determinize,
                       "word_determinize: if true, do a second pass "
                       "determinization on words only.")
        .def_readwrite(
            "minimize", &PyClass::minimize,
            "minimize: if true, push and minimize after determinization.")
        .def("__str__", [](const PyClass& opts) {
          std::ostringstream os;
          os << "delta: " << opts.delta << "\n";
          os << "max_mem: " << opts.max_mem << "\n";
          os << "phone_determinize: " << opts.phone_determinize << "\n";
          os << "word_determinize: " << opts.word_determinize << "\n";
          os << "minimize: " << opts.minimize << "\n";
          return os.str();
        });
  }
}
