// pybind/lat/kaldi_lattice_pybind.cc

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

#include "lat/kaldi_lattice_pybind.h"

#include "fst/arc_pybind.h"
#include "fst/fst_pybind.h"
#include "fst/vector_fst_pybind.h"
#include "fstext/lattice-weight.h"
#include "lat/kaldi-lattice.h"

using namespace kaldi;

void pybind_kaldi_lattice(py::module& m) {
  pybind_arc_impl<LatticeWeight>(m, "LatticeArc");
  pybind_vector_fst_impl<LatticeArc>(m, "Lattice");
  pybind_state_iterator_impl<Lattice>(m, "LatticeStateIterator");
  pybind_arc_iterator_impl<Lattice>(m, "LatticeArcIterator");
  pybind_mutable_arc_iterator_impl<Lattice>(m, "LatticeMutableArcIterator");

  pybind_arc_impl<CompactLatticeWeight>(m, "CompactLatticeArc");
  pybind_vector_fst_impl<CompactLatticeArc>(m, "CompactLattice");
  pybind_state_iterator_impl<CompactLattice>(m, "CompactLatticeStateIterator");
  pybind_arc_iterator_impl<CompactLattice>(m, "CompactLatticeArcIterator");
  pybind_mutable_arc_iterator_impl<CompactLattice>(
      m, "CompactLatticeMutableArcIterator");

  m.def("WriteLattice", &WriteLattice, py::arg("os"), py::arg("binary"),
        py::arg("lat"));

  m.def("WriteCompactLattice", &WriteCompactLattice, py::arg("os"),
        py::arg("binary"), py::arg("clat"));

  m.def("ReadLattice",
        [](std::istream& is, bool binary) -> Lattice* {
          Lattice* p = nullptr;
          bool ret = ReadLattice(is, binary, &p);
          if (!ret) {
            KALDI_ERR << "Failed to read lattice";
          }
          return p;
          // NOTE(fangjun): p points to a memory area allocated by `operator
          // new` we ask python to take the ownership of the allocate memory
          // which will finally calls `operator delete`
          //
          // Refer to
          // https://pybind11-rtdtest.readthedocs.io/en/stable/advanced.html
          // for the explanation of `return_value_policy::take_ownership`
        },
        py::arg("is"), py::arg("binary"),
        py::return_value_policy::take_ownership);

  m.def("ReadCompactLattice",
        [](std::istream& is, bool binary) -> CompactLattice* {
          CompactLattice* p = nullptr;
          bool ret = ReadCompactLattice(is, binary, &p);
          if (!ret) {
            KALDI_ERR << "Failed to read compact lattice";
          }
          return p;
        },
        py::arg("is"), py::arg("binary"),
        py::return_value_policy::take_ownership);
}
